# pylint: disable=W1203
"""
Script to check audio model stats
"""

# TODO: refactor

from datetime import datetime
import logging
import os
from pathlib import Path
from timeit import default_timer
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.noisemaker_iter_dataset import NoiseMakerIterDataset
from models import audio as audio_models


dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 4096
BATCH_SIZE = 64
NUM_ITERS = int(np.floor(NUM_SAMPLES / BATCH_SIZE))

DATASET = "NoiseMakerDataset"

RENDER_DURATION_IN_SEC = 4.0
MIDI_NOTE = None  # 60
MIDI_VELOCITY = None  # 110
MIDI_DURATION_IN_SEC = 2.0
BASE_SEED = 54745

AUDIO_FE = [
    # "mel128",
    "mn04",
    # "mn04_mel",
    # "mn20",
    # "mn20_mel",
    # "openl3_mel256_music_6144",
    # "passt_s",
    # "audiomae_ctx8",
]

EXPORT_PATH = Path(os.environ["PROJECT_ROOT"]) / "div_check" / "audio_fe"
if not EXPORT_PATH.exists():
    EXPORT_PATH.mkdir(parents=True)

settings_dict = {
    "device": DEVICE,
    "dataset": DATASET,
    "num_samples": NUM_SAMPLES,
    "batch_size": BATCH_SIZE,
    "num_iters": NUM_ITERS,
    "render_duration_in_sec": RENDER_DURATION_IN_SEC,
    "midi_note": MIDI_NOTE,
    "midi_velocity": MIDI_VELOCITY,
    "midi_duration_in_sec": MIDI_DURATION_IN_SEC,
    "base_seed": BASE_SEED,
    "audio_fe": AUDIO_FE,
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXPORT_PATH / f"stats-audio-models_{DATASET}_{dt_string}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)


if __name__ == "__main__":
    for k, v in settings_dict.items():
        log.info(f"{k}: {v}")

    for m in AUDIO_FE:
        log.info("")
        audio_fe = getattr(audio_models, m)()
        audio_fe.to(DEVICE)
        audio_fe.eval()

        dataset = NoiseMakerIterDataset(
            sample_rate=audio_fe.sample_rate,
            render_duration_in_sec=RENDER_DURATION_IN_SEC,
            midi_note=MIDI_NOTE,
            midi_velocity=MIDI_VELOCITY,
            midi_duration_in_sec=MIDI_DURATION_IN_SEC,
            base_seed=BASE_SEED,
        )

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)
        pbar = tqdm(dataloader, total=NUM_ITERS, dynamic_ncols=True)
        audio_repr = []

        log.info(f"Audio feature extractor: {m}")
        start = default_timer()
        for i, (_, _, audio) in enumerate(pbar):
            with torch.no_grad():
                audio = audio.to(DEVICE)

            audio_repr.append(audio_fe(audio))

            if i == NUM_ITERS - 1:
                break

        exec_time = default_timer() - start
        audio_repr = torch.cat(audio_repr, dim=0)

        log.info(f"Execution time: {exec_time:.2f}s")
        log.info(f"Output shape per sample: {audio_repr.shape[-1]}")
        log.info(f"Mean: {audio_repr.mean()}")
        log.info(f"Std: {audio_repr.std()}")
