import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

MODEL_ID = "m-a-p/MERT-v1-95M"
proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)

B = 2
in_sr = 16000
dur_s = 1.0
T = int(in_sr * dur_s)

# CPU에서 oracle 비교하는 게 제일 깔끔함
x = torch.randn(B, T, device="cpu")

target_sr = int(proc.sampling_rate)
print("processor sampling_rate:", target_sr)
print("do_normalize:", getattr(proc, "do_normalize", None))
print("padding_side:", getattr(proc, "padding_side", None))
print("padding_value:", getattr(proc, "padding_value", None))
print("return_attention_mask:", getattr(proc, "return_attention_mask", None))

# 1) 먼저 torch로 SR 맞추기 (16k -> 24k)
if in_sr != target_sr:
    resampler = torchaudio.transforms.Resample(in_sr, target_sr)
    x_rs = resampler(x)  # (B, 24000)
else:
    x_rs = x

# 2) HF processor oracle: "이미 24k인 입력"을 넣는다
#    (processor는 resample 안 하므로, 여기서 x_rs를 넣는게 핵심!)
x_list = [t.numpy() for t in x_rs]  # numpy list
oracle = proc(
    x_list,
    sampling_rate=target_sr,   # 메타로 알려줌
    return_tensors="pt",
    padding=True
)["input_values"]  # (B, 24000)

# 3) torch-only normalize 후보 (per-utterance zero-mean / unit-variance)
mean = x_rs.mean(dim=1, keepdim=True)
var = (x_rs - mean).pow(2).mean(dim=1, keepdim=True)
x_norm = (x_rs - mean) / torch.sqrt(var + 1e-5)

print("oracle shape:", oracle.shape, "torch shape:", x_norm.shape)

# 4) 차이 측정
l2 = (oracle - x_norm).pow(2).mean().item()
mx = (oracle - x_norm).abs().max().item()
print("L2 diff mean:", l2)
print("max abs diff:", mx)

# 5) 부가 체크: oracle 통계가 정말 표준화 형태인지
print("oracle mean/std per sample:")
print("  mean:", oracle.mean(dim=1))
print("  std :", oracle.std(dim=1, unbiased=False))
