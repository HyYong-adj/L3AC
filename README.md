# L3AC (Train Branch – MTG-Jamendo Weights)

This repository contains a customized L3AC architecture (train branch) and utilities to load pretrained weights hosted on Hugging Face.

* Code (architecture, training):
  https://github.com/HyYong-adj/L3AC/tree/train
* Pretrained weights (MTG-Jamendo):
  https://huggingface.co/choihy/mtg-l3ac

This design follows the upstream L3AC philosophy:

* Code is installed via pip
* Model weights are downloaded on demand and cached locally

## Installation
### 1) Install directly from GitHub (recommended)

You can install this train branch directly using pip:
```bash
pip install "git+https://github.com/HyYong-adj/L3AC.git@train"
```

For development or modification:
```bash
git clone -b train https://github.com/HyYong-adj/L3AC.git
cd L3AC
pip install -e .
```

⚠️ If you already have the official l3ac installed from PyPI, uninstall it first to avoid conflicts:

```bash
pip uninstall -y l3ac
```

### 2) Optional dependency (demo)

To run the demo example below, install librosa:

pip install librosa

```bash
Quickstart: Encode & Decode Audio
import librosa
import torch
import l3ac

# List available model configs (from l3ac/configs/*.toml)
print("Available models:", l3ac.list_models())

MODEL_USED = "1kbps_music"   # example config name
codec = l3ac.get_model(MODEL_USED)

print("Loaded model:", MODEL_USED)
print("Codec sample rate:", codec.config.sample_rate)

# Load example audio
sample_audio, sample_rate = librosa.load(librosa.example("libri1"))
sample_audio = sample_audio[None, :]
sample_audio = librosa.resample(
    sample_audio,
    orig_sr=sample_rate,
    target_sr=codec.config.sample_rate
)

codec.network.cuda()
codec.network.eval()

with torch.inference_mode():
    audio_in = torch.tensor(sample_audio, dtype=torch.float32, device="cuda")
    _, audio_length = audio_in.shape

    q_feature, indices = codec.encode_audio(audio_in)
    audio_out = codec.decode_audio(q_feature)
    generated_audio = audio_out[:, :audio_length].detach().cpu().numpy()

mse = ((sample_audio - generated_audio) ** 2).mean().item()
print(f"MSE: {mse}")
```
----
## Pretrained Weights (Hugging Face)

Pretrained weights are hosted on Hugging Face:

👉 https://huggingface.co/choihy/mtg-l3ac

### Weight download behavior

* When get_model() is called:

1. The config (.toml) specifies model_name and model_version
2. The corresponding weights are downloaded from Hugging Face
3. Weights are cached locally
4. Subsequent runs reuse the cached weights

### Default cache location
```text
~/.cache/l3ac/<model_name>.<model_version>/
```
If the weights already exist, downloading is skipped.

----

## Hugging Face Repository Structure

The Hugging Face model repository follows this layout:

```text
choihy/mtg-l3ac
├─ README.md
└─ weights/
   └─ <model_name>.<model_version>/
      ├─ encoder.pt
      ├─ decoder.pt
      ├─ quantizer.pt
      └─ ...
```

Each .pt file corresponds to one trainable module defined in the L3AC network.

----

## Model Configuration

Model configurations are stored in:
```text
l3ac/configs/*.toml
```

Example (1kbps_music.toml):
```toml
model_name = "mtg_l3ac"
model_version = "1kbps"
sample_rate = 16000

[network_config]
# architecture definition
```

If weight_url is **not explicitly set**, the code automatically resolves the URL to the Hugging Face repository.

----
## Using Your Own Trained Weights
### Option A) Upload to Hugging Face (recommended)

1. Upload your trained weights to:
```arduino
https://huggingface.co/choihy/mtg-l3ac
```
2. Follow the directory structure shown above
3. Set model_name and model_version in the config
4. Call:
```python
codec = l3ac.get_model("<config_name>")
```
No code changes are required.

----
### Option B) Use local weights (offline)

Place your module weights manually into:
'''python
~/.cache/l3ac/<model_name>.<model_version>/
```
Then run:
```python
codec = l3ac.get_model("<config_name>")
```
----
## Reproducibility Tips

* For stable experiments, pin a commit hash:
````bash
pip install "git+https://github.com/HyYong-adj/L3AC.git@<COMMIT_SHA>"
```
* Keep model_name + model_version immutable once published
----
## Acknowledgements

This repository is based on the original L3AC project and extends it for research on streaming neural audio codecs and music-domain training (MTG-Jamendo).


### available models

| config_name | Sample rate(Hz) | tokens/s | Codebook size | Bitrate(bps) |
|-------------|-----------------|----------|---------------|--------------|
| 0k75bps     | 16,000          | 44.44    | 117,649       | 748.6        |
| 1kbps       | 16,000          | 59.26    | 117,649       | 998.2        |
| 1k5bps      | 16,000          | 88.89    | 117,649       | 1497.3       |
| 3kbps       | 16,000          | 166.67   | 250,047       | 2988.6       |
