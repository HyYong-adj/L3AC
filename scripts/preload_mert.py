"""Preload MERT model and processor into HF cache and optional device.

Usage:
    python scripts/preload_mert.py --model m-a-p/MERT-v1-95M --device cuda

This script will download the model and processor and (optionally) move the model
onto the requested device to warm up caches and avoid first-run overhead during
training.
"""
import argparse
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='m-a-p/MERT-v1-95M')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading processor for {args.model}...")
    proc = Wav2Vec2FeatureExtractor.from_pretrained(args.model, trust_remote_code=True)
    print('Processor loaded, sample_rate=', getattr(proc, 'sampling_rate', None))

    print(f"Loading model {args.model} (this may take a while)...")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    print('Model loaded; moving to device', args.device)
    try:
        model.to(torch.device(args.device))
    except Exception as e:
        print('Warning: moving model to device failed:', e)

    print('Model and processor are cached. Exiting.')

if __name__ == '__main__':
    main()
