# tools/loss/mert_test.py
import os
import sys
import math
from dataclasses import dataclass

import torch
import torchaudio

from transformers import AutoModel, Wav2Vec2FeatureExtractor


MODEL_ID = os.environ.get("MERT_MODEL_ID", "m-a-p/MERT-v1-95M")

# If your input audio is not 16k, set it here or via env
IN_SR = int(os.environ.get("IN_SR", "16000"))

# seconds of fake audio for tests
DUR_S = float(os.environ.get("DUR_S", "1.0"))

# batch size
B = int(os.environ.get("B", "2"))

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def layer_norm_1d(wav: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Processor config says do_normalize=True.
    Wav2Vec2FeatureExtractor typically does per-utterance zero-mean/unit-variance.
    wav: (B,T)
    """
    mean = wav.mean(dim=1, keepdim=True)
    var = (wav - mean).pow(2).mean(dim=1, keepdim=True)
    return (wav - mean) / torch.sqrt(var + eps)


def torch_preprocess_like_processor(
    wav: torch.Tensor,
    in_sr: int,
    target_sr: int,
    do_normalize: bool = True,
    padding_value: float = 0.0,
    return_attention_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Torch-only equivalent of key processor behaviors:
      - resample to target_sr
      - (optional) per-utterance normalize
      - padding_value=0, padding_side='right' (here we keep same length so no padding needed)
      - return_attention_mask=True -> ones mask
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    assert wav.dim() == 2, f"expected (B,T), got {wav.shape}"

    # Resample: keep grad
    if in_sr != target_sr:
        resampler = torchaudio.transforms.Resample(in_sr, target_sr)
        # try to run on same device; fall back to CPU if needed
        try:
            resampler = resampler.to(wav.device)
            wav = resampler(wav)
        except Exception as e:
            if wav.is_cuda:
                print(f"[WARN] CUDA resample failed ({e}); CPU fallback.")
                wav_cpu = wav.to("cpu")
                resampler = resampler.to("cpu")
                wav = resampler(wav_cpu).to(wav.device)
            else:
                raise

    # Padding: in our test we keep fixed length, so no padding needed.
    # But we respect padding_value by ensuring zeros are zeros if any exist.
    if padding_value != 0.0:
        # rarely used; here just a placeholder
        pass

    if do_normalize:
        wav = layer_norm_1d(wav)

    attn = None
    if return_attention_mask:
        attn = torch.ones(wav.shape[0], wav.shape[1], device=wav.device, dtype=torch.long)

    return wav, attn


def main():
    torch.set_printoptions(sci_mode=False, precision=6)

    print(f"MODEL_ID: {MODEL_ID}")
    print(f"DEVICE: {DEVICE}")
    print(f"IN_SR: {IN_SR}, DUR_S: {DUR_S}, B: {B}")

    # Load processor just for config verification / SR
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("\n[Processor config]")
    print(f"  sampling_rate: {processor.sampling_rate}")
    print(f"  do_normalize: {getattr(processor, 'do_normalize', None)}")
    print(f"  padding_side: {getattr(processor, 'padding_side', None)}")
    print(f"  padding_value: {getattr(processor, 'padding_value', None)}")
    print(f"  return_attention_mask: {getattr(processor, 'return_attention_mask', None)}")

    target_sr = int(processor.sampling_rate)

    # Load model
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Print model forward signature keys (sanity)
    import inspect
    sig = inspect.signature(model.forward)
    print("\n[Model forward signature]")
    print(sig)

    # ------------------------------------------------------------
    # Test 1: confirm HF processor breaks grad (expected)
    # ------------------------------------------------------------
    print("\n[Test 1] HF processor grad break check (expected: requires_grad=False)")
    T_in = int(round(IN_SR * DUR_S))
    x = torch.randn(B, T_in, device=DEVICE, requires_grad=True)

    # HF processor path (will detach / recreate tensor)
    inputs_hf = processor(
        [t.detach().cpu().numpy() for t in x],  # mimic common usage
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    )
    y = inputs_hf["input_values"]
    print(f"  y.requires_grad: {y.requires_grad}")
    print(f"  y.grad_fn: {y.grad_fn}")
    print(f"  y.shape: {tuple(y.shape)}")

    # ------------------------------------------------------------
    # Test 2: torch-only preprocess keeps grad (MUST pass)
    # ------------------------------------------------------------
    print("\n[Test 2] Torch-only preprocess keeps grad (expected: requires_grad=True, grad_fn!=None)")
    wav2, attn2 = torch_preprocess_like_processor(
        x,
        in_sr=IN_SR,
        target_sr=target_sr,
        do_normalize=True,
        padding_value=0.0,
        return_attention_mask=True,
    )
    print(f"  wav2.requires_grad: {wav2.requires_grad}")
    print(f"  wav2.grad_fn: {wav2.grad_fn}")
    print(f"  wav2.shape: {tuple(wav2.shape)}")
    print(f"  attn2 is None?: {attn2 is None}, attn2.shape: {None if attn2 is None else tuple(attn2.shape)}")

    # ------------------------------------------------------------
    # Test 3: model forward w/ torch-only inputs and hidden_states
    # ------------------------------------------------------------
    print("\n[Test 3] Model forward + hidden_states shape check")
    # Many wav2vec2-like models accept input_values/attention_mask
    kwargs = {}
    if "input_values" in sig.parameters:
        kwargs["input_values"] = wav2
    else:
        # fallback to first positional arg name if remote code differs
        first = list(sig.parameters.keys())[0]
        kwargs[first] = wav2

    if attn2 is not None and "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = attn2

    kwargs["output_hidden_states"] = True

    outputs = model(**kwargs)
    hs = outputs.hidden_states
    assert hs is not None, "hidden_states is None; ensure output_hidden_states=True is supported"
    print(f"  num_hidden_states: {len(hs)}")
    print(f"  hidden_states[0].shape: {tuple(hs[0].shape)}  (expected: (B, T', C))")
    print(f"  hidden_states[-1].shape: {tuple(hs[-1].shape)}")

    stacked = torch.stack(list(hs), dim=0)  # (L,B,T',C)
    print(f"  stacked.shape: {tuple(stacked.shape)}  (expected: (L, B, T', C))")

    # ------------------------------------------------------------
    # Test 4: differentiability end-to-end (MUST pass)
    # ------------------------------------------------------------
    print("\n[Test 4] End-to-end backward to x (expected: x.grad != None and nonzero)")
    # simple combined feature: last layer mean
    feat = hs[-1]  # (B,T',C)
    loss = feat.mean()
    loss.backward(retain_graph=True)

    x_grad = x.grad
    print(f"  x.grad is None?: {x_grad is None}")
    if x_grad is not None:
        print(f"  x.grad.abs().mean(): {x_grad.abs().mean().item():.8f}")
        print(f"  x.grad.abs().max():  {x_grad.abs().max().item():.8f}")

    # ------------------------------------------------------------
    # Test 5: layer-weighted aggregation sanity (optional)
    # ------------------------------------------------------------
    print("\n[Test 5] Layer-weighted aggregation sanity")
    L = stacked.shape[0]
    layer_weights = torch.nn.Parameter(torch.ones(L, device=stacked.device))
    w = torch.softmax(layer_weights, dim=0).to(stacked.dtype)
    combined = (w.view(L, 1, 1, 1) * stacked).sum(0)  # (B,T',C)
    print(f"  combined.shape: {tuple(combined.shape)}")

    # optional: gradient through weights? (weights should get grad)
    (combined.mean()).backward()
    print(f"  layer_weights.grad is None?: {layer_weights.grad is None}")
    if layer_weights.grad is not None:
        print(f"  layer_weights.grad.abs().mean(): {layer_weights.grad.abs().mean().item():.8f}")

    print("\n✅ Done. If Test 4 passes, your MertPerceptualLoss can backprop to generator outputs.")


if __name__ == "__main__":
    main()
