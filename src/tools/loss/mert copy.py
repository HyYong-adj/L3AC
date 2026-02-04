import pathlib

import torch
import torch.nn.functional as F
import torchaudio

import utils
import xtract.nn as xnn

log = utils.log.get_logger()


class MertPerceptualLoss(torch.nn.Module):

    """MERT-based perceptual loss using Hugging Face MERT model.

    Behaviour:
      - If `model_path` is a HF model id or local checkpoint, try to load
        AutoModel + Wav2Vec2FeatureExtractor (trust_remote_code=True).
      - If transformers are not available or loading fails, fall back to the
        prototype mel+conv extractor (backward-compatible with earlier code).

    Features returned have shape (batch, channels, time) to match existing
    PerceptualLoss semantics.

    Args:
        model_path: HF model id (e.g. 'm-a-p/MERT-v1-95M') or local path. If None,
                    prototype extractor will be used.
        sample_rate: input audio sample rate (will be resampled to model's rate)
        reduction_func: function used to compute loss between two feature maps
        use_layer_weights: whether to use a learnable weighted average across
                           MERT layers (defaults to True)
    """

    def __init__(self, model_path: pathlib.Path | str | None = 'm-a-p/MERT-v1-95M', sample_rate=16000,
                 reduction_func=torch.nn.MSELoss(reduction='none'), use_layer_weights=True, device: str | None = 'auto'):
        super().__init__()
        self.requested_sample_rate = sample_rate
        self.reduction_func = reduction_func
        self.use_layer_weights = use_layer_weights

        # device: 'auto' selects CUDA if available, otherwise CPU; or pass 'cpu'/'cuda:0'
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

        # Transformer-based MERT model and processor (if available)
        self._hf_model = None
        self._processor = None
        self._num_layers = None
        self.layer_weights = None

        try:
            # Import HF utilities lazily
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            # Normalize model_path: accept None, empty string, or 'None' (string) as disabling HF MERT
            if isinstance(model_path, str) and model_path.strip().lower() in ('', 'none'):
                model_path = None

            # If model_path is None, the caller explicitly requested the prototype extractor
            if model_path is None:
                log.info("model_path is None; using prototype feature extractor")
                self._hf_model = None
                self._processor = None
            else:
                model_id = str(model_path)

                # Load processor and model (trust_remote_code required for MERT)
                try:
                    self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
                    self._hf_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
                    # freeze model params but allow gradients to flow to inputs
                    xnn.freeze(self._hf_model)

                    # Move model to the requested device (GPU if available)
                    try:
                        self._hf_model.to(self.device)
                    except Exception:
                        log.warning(f"Could not move MERT model to device {self.device}; continuing on its current device")

                    # set model to eval
                    self._hf_model.eval()

                    self._model_device = next(self._hf_model.parameters()).device

                    # Determine expected sampling rate
                    self._model_sample_rate = getattr(self._processor, 'sampling_rate', 16000)

                    # Determine number of hidden layers robustly.
                    # Prefer config.num_hidden_layers; otherwise perform a tiny forward pass to infer.
                    self._num_layers = None
                    try:
                        n_hidden = int(getattr(self._hf_model.config, 'num_hidden_layers'))
                        self._num_layers = n_hidden + 1
                    except Exception:
                        self._num_layers = None

                    if self._num_layers is None and self.use_layer_weights:
                        try:
                            with torch.no_grad():
                                probe_len = max(1, int(self._model_sample_rate * 0.1))  # 100ms probe
                                probe = torch.zeros(1, probe_len)
                                proc_inputs = self._processor([probe.numpy()], sampling_rate=self._model_sample_rate, return_tensors='pt', padding=True)
                                for k, v in proc_inputs.items():
                                    if isinstance(v, torch.Tensor):
                                        proc_inputs[k] = v.to(self._model_device)
                                outputs = self._hf_model(**proc_inputs, output_hidden_states=True)
                                self._num_layers = len(outputs.hidden_states)
                        except Exception as e:
                            log.warning(f"Could not infer MERT hidden layer count from a probe forward: {e}")
                            self._num_layers = 13

                    # Prepare layer weights if requested and we now have a layer count
                    if self.use_layer_weights:
                        if self._num_layers is None:
                            self._num_layers = 13
                        w = torch.ones(self._num_layers, device=self._model_device)
                        self.layer_weights = torch.nn.Parameter(w)
                        # register so it moves with model/device
                        self.register_parameter('layer_weights', self.layer_weights)

                    log.info(f"Loaded HF MERT model {model_id} with {_fmt_layers(self._num_layers)} layers on {self._model_device}")
                except Exception as e:
                    log.info(f"Failed to load HF MERT model '{model_id}': {e}; falling back to prototype extractor")
                    self._hf_model = None
                    self._processor = None
        except Exception:
            log.info("transformers not available; using prototype feature extractor")

        # Prototype feature extractor (used when no HF MERT model is available)
        if self._hf_model is None:
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, power=1)
            self.prototype_proj = torch.nn.Sequential(
                torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(256, 512, kernel_size=3, padding=1)
            )

    @staticmethod
    def _ensure_list_of_numpy(audios: torch.Tensor):
        # convert batch tensor to list of 1D numpy arrays as HF processor expects
        audios = audios.detach().cpu()
        if audios.ndim == 1:
            return [audios.numpy()]
        return [a.numpy() for a in audios]

    def forward(self, predictions, targets):
        predict_features = self.get_feature(predictions)
        target_features = self.get_feature(targets)
        # Keep behavior consistent with Whisper-based loss in asr.py
        # features are expected to have shape (batch, channels, time)
        return self.reduction_func(predict_features, target_features).sum(2).mean()

    def get_feature(self, audios: torch.Tensor):
        # Resample to the requested input sampling rate first
        audios = self.resampler(audios)

        # If HF MERT is available, use it for features
        if self._hf_model is not None and self._processor is not None:
            # Build list of CPU numpy arrays for the processor (HF processor pads using numpy)
            batch_list = []
            if isinstance(audios, torch.Tensor):
                if audios.dim() == 1:
                    tensors = [audios]
                else:
                    tensors = [a for a in audios]
                for t in tensors:
                    batch_list.append(t.detach().cpu().numpy())
            else:
                # audios may already be a list/sequence of numpy arrays or tensors
                batch_list = [ (a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a) for a in audios ]

            inputs = self._processor(batch_list, sampling_rate=self._model_sample_rate,
                                      return_tensors='pt', padding=True)

            # Move inputs to model device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._model_device)

            # Run model (do NOT disable grad; generator should receive gradients through features)
            outputs = self._hf_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # list of (batch, seq_len, dim)

            # stack -> (layers, batch, seq_len, dim)
            stacked = torch.stack(hidden_states, dim=0).to(self._model_device)

            if self.use_layer_weights:
                L = stacked.shape[0]
                # If a registered parameter exists and matches layer count, use it
                if (self.layer_weights is not None) and (self.layer_weights.numel() == L):
                    w = F.softmax(self.layer_weights.to(stacked.device), dim=0).to(stacked.dtype)  # (L,)
                else:
                    # Fallback: create uniform weights (non-learnable) and warn once
                    log.warning(f"MERT layer_weights missing or size mismatch (expected {L}, got {None if self.layer_weights is None else self.layer_weights.numel()}). Using uniform weights.")
                    w = torch.ones(L, device=stacked.device, dtype=stacked.dtype) / float(L)

                # Weighted sum over layer dimension L using broadcasting to avoid einsum pitfalls
                try:
                    combined = (w.view(L, 1, 1, 1) * stacked).sum(0)  # -> (batch, seq_len, dim)
                except Exception as e:
                    log.error(f"Failed to combine MERT layers: w.shape={w.shape}, stacked.shape={stacked.shape}, error={e}")
                    raise
            else:
                # default to last layer
                combined = stacked[-1]

            # return shape (batch, channels, time) i.e., (batch, dim, seq_len)
            return combined.permute(0, 2, 1).contiguous()

        # Prototype: log-mel + conv projection
        # Ensure audios has batch dim: (batch, time)
        if audios.dim() == 1:
            audios = audios.unsqueeze(0)
        tensor_mel = self.mel(audios).clamp(min=1e-9).log()
        # torchaudio may return (n_mels, time) for single example; ensure batch dim present
        if tensor_mel.dim() == 2:
            tensor_mel = tensor_mel.unsqueeze(0)
        # Now tensor_mel should be (batch, n_mels, time) which Conv1d expects as (N, C, L)
        features = self.prototype_proj(tensor_mel)
        return features

    def get_results_from_features(self, features):
        # HF MERT does not necessarily provide a decode API like Whisper; return placeholders
        if self._hf_model is not None:
            try:
                # try a generic decode if present
                results = self._hf_model.decode(features)
                return [r.text for r in results]
            except Exception:
                log.warning("MERT model decode not available; returning placeholders")
        return ["<mert-placeholder>" for _ in range(features.shape[0])]


def _fmt_layers(n):
    return f"{n} hidden" if n is not None else "? hidden"
