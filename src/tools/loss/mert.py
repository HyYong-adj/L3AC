# tools/loss/mert.py
"""
MERT perceptual loss for TRAINING (gradient flows to generator outputs).

Verified equivalence (by oracle test):
- processor.sampling_rate = 24000
- processor.do_normalize = True  -> per-utterance zero-mean / unit-variance (eps=1e-5)
- padding_value = 0, padding_side = right, return_attention_mask = True
- IMPORTANT: HF processor call breaks autograd, so we replicate its behavior in torch.

Loss behavior (matches Whisper-style):
    loss = reduction_func(feat_pred, feat_tgt).sum(-1).mean()
where features are shaped (B, T', C).

Notes:
- This module loads HF processor ONLY to read metadata (sampling_rate, do_normalize, etc.).
  It NEVER calls processor(...) on waveforms.
- output_hidden_states=True is expensive. You can:
    * set use_layer_weights=False to use last hidden state only, OR
    * set layer_subset to reduce cost (e.g., (-4, -3, -2, -1)).
"""

from __future__ import annotations

import pathlib
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torchaudio

import utils
import xtract.nn as xnn

log = utils.log.get_logger()


class MertPerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        model_id_or_path: Union[str, pathlib.Path] = "m-a-p/MERT-v1-95M",
        input_sample_rate: int = 16000,
        reduction_func=torch.nn.MSELoss(reduction="none"),
        # Layer aggregation
        use_layer_weights: bool = True,
        layer_subset: Optional[Sequence[int]] = None,  # e.g. (-4,-3,-2,-1)
        # Device / precision
        device: Union[str, torch.device, None] = "auto",
        amp_dtype: Optional[torch.dtype] = None,  # e.g. torch.bfloat16 / torch.float16
    ):
        super().__init__()
        self.model_id_or_path = str(model_id_or_path)
        self.input_sample_rate = int(input_sample_rate)
        self.reduction_func = reduction_func
        self.use_layer_weights = bool(use_layer_weights)
        self.layer_subset = tuple(layer_subset) if layer_subset is not None else None
        self.amp_dtype = amp_dtype

        # device
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load HF components
        try:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers is required for MertPerceptualLoss: {e}")

        # Read processor config ONLY (safe)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        self.model_sample_rate = int(getattr(proc, "sampling_rate", 24000))
        self.do_normalize = bool(getattr(proc, "do_normalize", True))
        self.padding_value = float(getattr(proc, "padding_value", 0.0))
        self.padding_side = str(getattr(proc, "padding_side", "right"))
        self.return_attention_mask = bool(getattr(proc, "return_attention_mask", True))

        if self.padding_side != "right":
            # MERT config we saw is "right"; if this changes, implement left padding mask logic.
            log.warning(f"padding_side={self.padding_side} (expected 'right'); mask/padding semantics may differ.")

        # Model
        self.model = AutoModel.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        xnn.freeze(self.model)
        self.model.eval()
        self.model.to(self.device)

        # Determine number of hidden states (for layer weights)
        self._num_hidden_states: Optional[int] = None
        try:
            n_hidden = int(getattr(self.model.config, "num_hidden_layers"))
            self._num_hidden_states = n_hidden + 1  # embeddings + each layer
        except Exception:
            self._num_hidden_states = None

        # Setup layer weights length
        if self.use_layer_weights:
            if self.layer_subset is not None:
                w_len = len(self.layer_subset)
            else:
                # fallback for MERT-v1-95M: 13 hidden states total
                w_len = int(self._num_hidden_states) if self._num_hidden_states is not None else 13
                if self._num_hidden_states is None:
                    log.warning("Could not infer num_hidden_layers; assuming 13 hidden_states for weights.")
            self.layer_weights = torch.nn.Parameter(torch.ones(w_len, device=self.device))
        else:
            self.layer_weights = None

        # Resampler input_sr -> model_sr
        self._resampler = torchaudio.transforms.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=self.model_sample_rate,
        )

        log.info(
            f"[MertPerceptualLoss] model={self.model_id_or_path} device={self.device} "
            f"input_sr={self.input_sample_rate} -> model_sr={self.model_sample_rate} "
            f"do_normalize={self.do_normalize} padding_value={self.padding_value} "
            f"return_attention_mask={self.return_attention_mask} "
            f"use_layer_weights={self.use_layer_weights} layer_subset={self.layer_subset}"
        )

    # -------------------------
    # Main loss
    # -------------------------
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_feat = self.get_feature(predictions)
        tgt_feat = self.get_feature(targets)

        # (B,T',C) -> sum over C, mean over B,T'
        loss_map = self.reduction_func(pred_feat, tgt_feat)
        return loss_map.sum(-1).mean()

    # -------------------------
    # Feature extraction
    # -------------------------
    def get_feature(self, audios: torch.Tensor) -> torch.Tensor:
        """
        audios: (B,T) or (T,)
        returns: (B,T',C)
        """
        if audios.dim() == 1:
            audios = audios.unsqueeze(0)
        if audios.dim() != 2:
            raise ValueError(f"Expected audios of shape (B,T) or (T,), got {tuple(audios.shape)}")

        # 1) resample to model SR (keep grad)
        audios = self._resample_preserve_grad(audios)

        # 2) move to model device (keep grad)
        wav = audios.to(self.device)

        # 3) processor-equivalent normalize (verified)
        if self.do_normalize:
            wav = self._normalize_like_processor(wav)

        # 4) attention mask (right padding semantics)
        # In your training, audio is fixed-length, so mask=1 is equivalent to processor behavior.
        attn_mask = None
        if self.return_attention_mask:
            attn_mask = torch.ones(wav.shape[0], wav.shape[1], device=wav.device, dtype=torch.long)

        # 5) MERT forward with hidden states
        # Model params are frozen; grad flows through wav.
        if self.amp_dtype is not None and wav.is_cuda:
            ctx = torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        else:
            ctx = _NullCtx()

        with ctx:
            outputs = self.model(
                input_values=wav,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("hidden_states is None. Ensure output_hidden_states=True is supported by the model.")

        # select layers
        if self.layer_subset is not None:
            hs = [hidden_states[i] for i in self.layer_subset]
        else:
            hs = list(hidden_states)

        stacked = torch.stack(hs, dim=0)  # (L,B,T',C)

        # combine layers
        if self.use_layer_weights:
            if self.layer_weights is None:
                raise RuntimeError("use_layer_weights=True but layer_weights is None")

            L = stacked.shape[0]
            if self.layer_weights.numel() != L:
                log.warning(
                    f"layer_weights mismatch: expected {L}, got {self.layer_weights.numel()}. Using uniform."
                )
                w = torch.ones(L, device=stacked.device, dtype=stacked.dtype) / float(L)
            else:
                w = F.softmax(self.layer_weights.to(stacked.device), dim=0).to(stacked.dtype)

            combined = (w.view(L, 1, 1, 1) * stacked).sum(0)  # (B,T',C)
        else:
            combined = stacked[-1]  # last layer (B,T',C)

        return combined.contiguous()

    # -------------------------
    # Processor-equivalent normalize (verified)
    # -------------------------
    @staticmethod
    def _normalize_like_processor(wav: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Matches processor do_normalize=True for MERT-v1-95M:
        per-utterance zero-mean / unit-variance normalization.
        wav: (B,T)
        """
        mean = wav.mean(dim=1, keepdim=True)
        var = (wav - mean).pow(2).mean(dim=1, keepdim=True)
        return (wav - mean) / torch.sqrt(var + eps)

    # -------------------------
    # Resample helper (keeps grad; CPU fallback if needed)
    # -------------------------
    def _resample_preserve_grad(self, audios: torch.Tensor) -> torch.Tensor:
        # rebuild if SR changed
        if int(self._resampler.orig_freq) != int(self.input_sample_rate) or int(self._resampler.new_freq) != int(self.model_sample_rate):
            self._resampler = torchaudio.transforms.Resample(self.input_sample_rate, self.model_sample_rate)

        try:
            self._resampler = self._resampler.to(audios.device)
            return self._resampler(audios)
        except Exception as e:
            if audios.is_cuda:
                log.warning(f"Resample on CUDA failed ({e}); using CPU fallback (slow).")
                cpu = audios.to("cpu")
                self._resampler = self._resampler.to("cpu")
                out = self._resampler(cpu)
                return out.to(audios.device)
            raise


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
