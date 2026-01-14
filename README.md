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
pip install accelerate datasets einops pynvml tensorboard
pip install pydantic-settings lz4 bidict
pip install scipy seaborn rich

pip install local-attention

pip install soundfile librosa
pip install openai-whisper pesq pystoi jiwer ptflops
```
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

## Overview
This repository extends the L3AC baseline by enforcing strict causality across the entire architecture.\
While the original paper claims causality, only the local transformer was causal and most convolutional operations exhibited future look-ahead (~100 ms).\
We replace all non-causal convolutions with **strict causal variants** and introduce **CausalGRNEMA**, a causal reformulation of ConvNeXt-V2’s GRN using EMA to avoid future leakage.\
This ensures the model is suitable for **streaming-safe** audio generation and coding.
