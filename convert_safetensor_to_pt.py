#!/usr/bin/env python3
"""
SafeTensor 파일들을 모아서 PyTorch .pt 형식으로 변환하는 스크립트
"""
import os
import glob
import torch
from safetensors.torch import load_file
from pathlib import Path


def convert_safetensors_to_pt(
    state_cache_dir: str,
    output_path: str,
    pattern: str = "model*.safetensors"
):
    """
    여러 safetensor 파일을 병합하여 단일 .pt 파일로 저장
    
    Args:
        state_cache_dir: safetensor 파일들이 있는 디렉토리
        output_path: 저장할 .pt 파일 경로
        pattern: 병합할 파일 패턴 (기본: "model*.safetensors")
    """
    state_cache_dir = Path(state_cache_dir)
    
    # safetensor 파일들 찾기
    safetensor_files = sorted(glob.glob(str(state_cache_dir / pattern)))
    
    if not safetensor_files:
        print(f"❌ No files found matching pattern: {pattern} in {state_cache_dir}")
        return
    
    print(f"📂 Found {len(safetensor_files)} safetensor files:")
    for f in safetensor_files:
        print(f"  - {Path(f).name}")
    
    # 모든 safetensor 파일 병합
    merged_state = {}
    for path in safetensor_files:
        print(f"\n⏳ Loading {Path(path).name}...")
        state = load_file(path, device="cpu")
        print(f"   Keys: {len(state)}, Total params: {sum(v.numel() for v in state.values()):,}")
        merged_state.update(state)
    
    print(f"\n✅ Merged state_dict:")
    print(f"   Total keys: {len(merged_state)}")
    print(f"   Total parameters: {sum(v.numel() for v in merged_state.values()):,}")
    
    # .pt 파일로 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    torch.save(merged_state, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✨ Done! File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    # usage example:
    # python convert_safetensor_to_pt.py --run-name 60119071448_UX5E
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SafeTensor files to PyTorch .pt format")
    parser.add_argument(
        "--run-name",
        type=str,
        default="60119071448_UX5E",
        help="WandB run name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/checkpoint",
        help="Output directory for .pt file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="model*.safetensors",
        help="File pattern to match (default: model*.safetensors)"
    )
    
    args = parser.parse_args()
    
    # input_dir 조합
    input_dir = f"/workspace/codec/L3AC/output/log/src.main.3kbps_music.{args.run_name}/state_cache"
    
    # output_path 조합
    output_path = Path(args.output_dir) / f"{args.run_name}2.pt"
    
    convert_safetensors_to_pt(
        state_cache_dir=input_dir,
        output_path=str(output_path),
        pattern=args.pattern
    )
