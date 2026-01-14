"""
Make XDataset from Jamendo MP3 tracks (no pre-cut clip files):
- download jamendo_all_mp3 dataset from https://github.com/MTG/mtg-jamendo-dataset
- Scan all *.mp3 under SOURCE_MP3_DIR
- Split by track (random, reproducible)
- For each track: load -> resample to 24k mono -> slice into 25s chunks -> yield samples
- Save XDataset to TARGET_DIR/{train,eval,test}

This matches your original code's semantics closely:
- DN.name: original track identifier (relative path)
- clip_idx: start sample index in the resampled waveform (NOT clip number)
- audio_duration: original track duration in seconds (based on resampled waveform length)
"""

from __future__ import annotations

import pathlib
import random

import pandas  # only used for convenience; can remove if you want
import runtime_resolver
import tools.audio
import utils
from xtract.data import x_dataset, x_feature
from prepare.data_process import DN

RS = runtime_resolver.init_runtime()
log = utils.log.get_logger()
RS = runtime_resolver.init_runtime()


# =========================
# Config (EDIT THESE)
# =========================
DATA_DIR = RS.data_path / "dataset"
print("DATA_DIR:", DATA_DIR)
# Folder that contains ALL Jamendo mp3 tracks (recursively)
# Example: DATA_DIR / "source" / "jamendo_all_mp3"
SOURCE_MP3_DIR = pathlib.Path("/data2/choihy/L3AC/mtg-jamendo-dataset/data")

# Where to save XDataset
TARGET_DIR = DATA_DIR / "mtg"
TARGET_DIR.mkdir(exist_ok=True, parents=True)

# Audio processing
DATASET_SAMPLE_RATE = 24000
DATASET_CHANNELS = 1
CLIP_SECONDS = 25.0

# Split
SEED = 42
TRAIN_RATIO = 0.90
EVAL_RATIO = 0.05
TEST_RATIO = 0.05

# Clip policy
KEEP_LAST_SHORT = True          # like your original generator (keeps last shorter-than-25s chunk)
MIN_LAST_SECONDS = 1.0          # drop extremely tiny tail clips if KEEP_LAST_SHORT=True

# XDataset audio storage hint (kept same as your original)
AUDIO_FMT = "MP3-medium"

# =========================
# XFeatures (same schema)
# =========================
x_features = {
    DN.name: x_feature.Value("string"),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
    "clip_idx": x_feature.Value("uint32"),
    "audio_duration": x_feature.Value("float32"),
}

audio_load_func = tools.audio.load


def list_all_mp3_paths(root: pathlib.Path) -> list[pathlib.Path]:
    mp3s = sorted([p for p in root.rglob("*.mp3") if p.is_file()])
    if not mp3s:
        raise FileNotFoundError(f"No mp3 files found under: {root}")
    return mp3s


def split_tracks(paths: list[pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    if abs((TRAIN_RATIO + EVAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(SEED)
    paths = paths[:]  # copy
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(n * TRAIN_RATIO)
    n_eval = int(n * EVAL_RATIO)
    train = paths[:n_train]
    eval_ = paths[n_train:n_train + n_eval]
    test = paths[n_train + n_eval:]
    return {"train": train, "eval": eval_, "test": test}


def rel_track_name(mp3_path: pathlib.Path) -> str:
    """
    Track identifier stored in DN.name.
    Use a stable relative path (like MTG TSV audio_name 'dir/file.mp3' style).
    """
    return mp3_path.relative_to(SOURCE_MP3_DIR).as_posix()


def iter_data_for_tracks(track_paths: list[pathlib.Path]):
    clip_len = int(CLIP_SECONDS * DATASET_SAMPLE_RATE)

    for mp3_path in track_paths:
        name = rel_track_name(mp3_path)

        # Load + resample + mono here, exactly like original (frame_rate=24000, channels=1)
        audio = audio_load_func(mp3_path, channels=DATASET_CHANNELS, frame_rate=DATASET_SAMPLE_RATE)

        # Duration based on resampled waveform length
        audio_duration = float(len(audio) / DATASET_SAMPLE_RATE)

        # Slice into 25s segments (sample-index exact, same as original)
        for idx in range(0, len(audio), clip_len):
            seg = audio[idx: idx + clip_len]

            if len(seg) == 0:
                continue

            # If last segment is short:
            if len(seg) < clip_len:
                if not KEEP_LAST_SHORT:
                    break
                if float(len(seg) / DATASET_SAMPLE_RATE) < MIN_LAST_SECONDS:
                    break

            yield {
                DN.name: name,
                DN.audio: seg,
                "clip_idx": int(idx),  # start sample index
                "audio_duration": audio_duration,
            }


def build_xdataset(track_paths: list[pathlib.Path]) -> x_dataset.XDataset:
    """
    Note: we pass a list of track paths to the generator via gen_kwargs.
    This matches your original from_generator usage pattern.
    """
    def _gen(paths):
        yield from iter_data_for_tracks(paths)

    ds = x_dataset.XDataset.from_generator(
        _gen,
        gen_kwargs=dict(paths=track_paths),
        x_features=x_features,
        num_proc=RS.cpu_num,
    )
    return ds


def init():
    log.info("=== Build XDataset from MP3 tracks (slice to 25s) ===")
    log.info(f"SOURCE_MP3_DIR: {SOURCE_MP3_DIR}")
    log.info(f"TARGET_DIR:     {TARGET_DIR}")
    log.info(f"sr={DATASET_SAMPLE_RATE}, ch={DATASET_CHANNELS}, clip={CLIP_SECONDS}s")
    log.info(f"split={TRAIN_RATIO}/{EVAL_RATIO}/{TEST_RATIO}, seed={SEED}")
    log.info(f"KEEP_LAST_SHORT={KEEP_LAST_SHORT}, MIN_LAST_SECONDS={MIN_LAST_SECONDS}")

    mp3_paths = list_all_mp3_paths(SOURCE_MP3_DIR)
    log.info(f"Found {len(mp3_paths)} tracks")

    splits = split_tracks(mp3_paths)
    for split_name in ("train", "eval", "test"):
        log.info(f"--- Building {split_name} dataset: {len(splits[split_name])} tracks ---")
        ds = build_xdataset(splits[split_name])
        out_dir = TARGET_DIR / split_name
        ds.save_to_disk(out_dir)
        log.info(f"Saved: {out_dir}")


if __name__ == "__main__":
    init()
