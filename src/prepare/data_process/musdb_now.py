"""
Build XDataset from MusDB wav tracks (resample and slice like mtg_now)
- Scans /data2/choihy/streaming/musdb/original recursively for *.wav
- Uses existing `train` and `test` folders: keeps `test` as test, splits `train` into train/eval by ratio
- For each WAV: load -> resample to 24k mono -> slice into 25s chunks -> yield samples
- Save XDataset to TARGET_DIR/{train,eval,test}

Usage:
    PYTHONPATH=src python -m prepare.data_process.musdb_now

"""

from __future__ import annotations

import pathlib
import random

import runtime_resolver
import tools.audio
import utils
from xtract.data import x_dataset, x_feature
from prepare.data_process import DN

RS = runtime_resolver.init_runtime()
log = utils.log.get_logger()

# =========================
# Config (EDIT THESE)
# =========================
DATA_DIR = RS.data_path / "dataset"
print("DATA_DIR:", DATA_DIR)
# Root that contains musdb original wavs with train/ test folders
SOURCE_WAV_DIR = pathlib.Path("/data2/choihy/streaming/musdb/original")

# Where to save XDataset
TARGET_DIR = DATA_DIR / "musdb"
TARGET_DIR.mkdir(exist_ok=True, parents=True)

# Audio processing
DATASET_SAMPLE_RATE = 24000
DATASET_CHANNELS = 1
CLIP_SECONDS = 25.0

# Split (train split will be split into train/eval according to TRAIN_RATIO and EVAL_RATIO)
SEED = 42
TRAIN_RATIO = 0.95  # fraction of original train set used as train (rest becomes eval)
EVAL_RATIO = 0.05
TEST_RATIO = 0.0  # test is taken from SOURCE_WAV_DIR/test entirely

# Clip policy
KEEP_LAST_SHORT = True
MIN_LAST_SECONDS = 1.0

# XDataset audio storage hint
AUDIO_FMT = "MP3-medium"

# =========================
# XFeatures
# =========================
x_features = {
    DN.name: x_feature.Value("string"),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
    "clip_idx": x_feature.Value("uint32"),
    "audio_duration": x_feature.Value("float32"),
}

audio_load_func = tools.audio.load


def list_all_wav_paths(root: pathlib.Path) -> list[pathlib.Path]:
    wavs = sorted([p for p in root.rglob("*.wav") if p.is_file()])
    if not wavs:
        raise FileNotFoundError(f"No wav files found under: {root}")
    return wavs


def rel_track_name(wav_path: pathlib.Path) -> str:
    return wav_path.relative_to(SOURCE_WAV_DIR).as_posix()


def split_train_eval(train_paths: list[pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    rng = random.Random(SEED)
    paths = train_paths[:]
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_RATIO)
    train = paths[:n_train]
    eval_ = paths[n_train:]
    return {"train": train, "eval": eval_}


def iter_data_for_tracks(track_paths: list[pathlib.Path]):
    clip_len = int(CLIP_SECONDS * DATASET_SAMPLE_RATE)

    for wav_path in track_paths:
        name = rel_track_name(wav_path)

        audio = audio_load_func(wav_path, channels=DATASET_CHANNELS, frame_rate=DATASET_SAMPLE_RATE)

        audio_duration = float(len(audio) / DATASET_SAMPLE_RATE)

        for idx in range(0, len(audio), clip_len):
            seg = audio[idx: idx + clip_len]

            if len(seg) == 0:
                continue

            if len(seg) < clip_len:
                if not KEEP_LAST_SHORT:
                    break
                if float(len(seg) / DATASET_SAMPLE_RATE) < MIN_LAST_SECONDS:
                    break

            yield {
                DN.name: name,
                DN.audio: seg,
                "clip_idx": int(idx),
                "audio_duration": audio_duration,
            }


def build_xdataset(track_paths: list[pathlib.Path]) -> x_dataset.XDataset:
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
    log.info("=== Build XDataset from MusDB wav tracks (slice to 25s) ===")
    log.info(f"SOURCE_WAV_DIR: {SOURCE_WAV_DIR}")
    log.info(f"TARGET_DIR:     {TARGET_DIR}")
    log.info(f"sr={DATASET_SAMPLE_RATE}, ch={DATASET_CHANNELS}, clip={CLIP_SECONDS}s")
    log.info(f"TRAIN_SPLIT={TRAIN_RATIO}, eval derived fraction={EVAL_RATIO}, seed={SEED}")
    log.info(f"KEEP_LAST_SHORT={KEEP_LAST_SHORT}, MIN_LAST_SECONDS={MIN_LAST_SECONDS}")

    train_dir = SOURCE_WAV_DIR / "train"
    test_dir = SOURCE_WAV_DIR / "test"

    train_paths = list_all_wav_paths(train_dir) if train_dir.exists() else []
    test_paths = list_all_wav_paths(test_dir) if test_dir.exists() else []

    log.info(f"Found {len(train_paths)} train tracks, {len(test_paths)} test tracks")

    splits = split_train_eval(train_paths)
    splits["test"] = test_paths

    for split_name in ("train", "eval", "test"):
        log.info(f"--- Building {split_name} dataset: {len(splits.get(split_name, []))} tracks ---")
        ds = build_xdataset(splits.get(split_name, []))
        out_dir = TARGET_DIR / split_name
        ds.save_to_disk(out_dir)
        log.info(f"Saved: {out_dir}")


if __name__ == "__main__":
    init()
