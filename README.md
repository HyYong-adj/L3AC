# L3AC_training
---
This is repository contains the training setup for strictly causal L3AC.


## Environment Setup

### 1) Create conda environment

```bash
conda create -n l3ac cuda=12.6 python=3.13 -c nvidia
conda activate l3ac
```
### 2) Install PyTorch (CUDA 12.6)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
### 3) Install dependencies
```bash
pip install transformers datasets accelerate einops
pip install accelerate datasets einops pynvml tensorboard
pip install pydantic-settings lz4 bidict
pip install scipy seaborn rich

pip install local-attention
pip install descript-audiotools

pip install soundfile librosa
pip install openai-whisper pesq pystoi jiwer ptflops
```

### MERT / HuggingFace Transformers (optional but recommended for MERT-based perceptual loss)

MERT models are provided through Hugging Face and require `transformers` + supporting packages. If you plan to use `MertPerceptualLoss` (default in some configs) or to load MERT models, install the following and ensure you have network access and sufficient disk space for model weights:

```bash
# core HF packages
pip install transformers datasets accelerate einops

# optional but useful for MERT and large-model usage
pip install tokenizers pynvml

# If you're using the repository's MERT-based loss, make sure transformers is available
# and that you have access to the HF model id (e.g. 'm-a-p/MERT-v1-95M'). For private models,
# set HF_TOKEN environment variable or run `huggingface-cli login`.

# Quick smoke test (run inside the `l3ac` env):
python - << 'PY'
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch

model_id = 'm-a-p/MERT-v1-95M'
proc = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
print('Loaded MERT model:', model.__class__.__name__)
PY
```

> Note: The HF MERT loader uses `trust_remote_code=True` because these models
> provide custom code. Only enable this for trusted models (official HF orgs)
> and ideally in an isolated environment.

### Preloading MERT (recommended)

Downloading large HF models during the first training run can add several
minutes to startup time and may fail late if there are network or device
configuration issues. To avoid that, we provide a small helper that downloads
and (optionally) moves the MERT model to a target device ahead of training.

Run the preload helper like this (example moves model to GPU 1):

```bash
python scripts/preload_mert.py --model m-a-p/MERT-v1-95M --device cuda:1
```

Notes:
- The command is optional but *recommended* for production runs to catch
  download or device placement issues early. ✅
- If you don't want to use HF MERT during training, set `mert_model_id = ''` or
  `mert_model_id = 'None'` in your config and the code will use the lightweight
  mel+conv prototype extractor instead.
- The preload script uses `trust_remote_code=True` like model loading; ensure
  you trust the source and run in an isolated environment if necessary.

### 4) DAC-related dependencies
```bash
pip install git+https://github.com/carlthome/audiotools.git@upgrade-dependencies
pip install descript-audio-codec
```
## Data Preparation

Data preprocessing scripts are located under:
```bash
./src/prepare/data_process
```

Follow the scripts in that directory to build your dataset / metadata.
```bash
# data prepare
see scripts in ./src/prepare/data_process
# in this repository we use mtg_now.py
```
## training model
```bash
# training model
accelerate launch --num_processes=1 $(pwd)/src/main.py --config 3kbps_music
#test eval
WANDB_DISABLED=true ONLY_EVAL=1 accelerate launch --num_processes=1 --mixed_precision bf16 $(pwd)/src/main.py --config 3kbps_music
```
| Adjust --config to match your available config names.


## install L3AC
```bash
cd L3AC
pip install -e .
```

## Overview
This repository extends the L3AC baseline by enforcing strict causality across the entire architecture.\
While the original paper claims causality, only the local transformer was causal and most convolutional operations exhibited future look-ahead (~100 ms).\
We replace all non-causal convolutions with **strict causal variants** and introduce **CausalGRNEMA**, a causal reformulation of ConvNeXt-V2’s GRN using EMA to avoid future leakage.\
This ensures the model is suitable for **streaming-safe** audio generation and coding.

