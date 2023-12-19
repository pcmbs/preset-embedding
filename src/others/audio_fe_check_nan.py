# pylint: disable=W1203
"""
Script to detect NaN in embeddings.
"""
# TODO: refactor

from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import SynthDataset
from utils.synth import PresetHelper
from models import audio as audio_models

dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_FE = "mn20"

NUM_SAMPLES = 2**13
BATCH_SIZE = 64
NUM_ITERS = int(np.floor(NUM_SAMPLES / BATCH_SIZE))

SYNTH = "tal_noisemaker"
RENDER_DURATION_IN_SEC = 4.0
MIDI_NOTE = None
MIDI_VELOCITY = None
MIDI_DURATION_IN_SEC = 1.0
BASE_SEED = 667

EXPORT_PATH = Path(os.environ["PROJECT_ROOT"]) / "div_check" / "audio_fe"
if not EXPORT_PATH.exists():
    EXPORT_PATH.mkdir(parents=True)

settings_dict = {
    "device": DEVICE,
    "audio_fe": AUDIO_FE,
    "num_samples": NUM_SAMPLES,
    "batch_size": BATCH_SIZE,
    "num_iters": NUM_ITERS,
    "render_duration_in_sec": RENDER_DURATION_IN_SEC,
    "midi_note": MIDI_NOTE,
    "midi_velocity": MIDI_VELOCITY,
    "midi_duration_in_sec": MIDI_DURATION_IN_SEC,
    "base_seed": BASE_SEED,
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXPORT_PATH / f"NaN_detector_{AUDIO_FE}_{SYNTH}_{dt_string}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)


if __name__ == "__main__":
    for k, v in settings_dict.items():
        log.info(f"{k}: {v}")

    audio_fe = getattr(audio_models, AUDIO_FE)()
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

    for i, (_, params, audio) in enumerate(pbar):
        audio = audio.to(DEVICE)
        with torch.no_grad():
            emb = audio_fe(audio)

        if torch.any(torch.isnan(emb)):
            nan_idxs = torch.where(torch.isnan(emb.mean(dim=-1)))[0]
            rms_nan = torch.sqrt(torch.mean(audio[nan_idxs] ** 2))
            log.info(f"NaN found in batch {i}")
            for j in nan_idxs:
                log.info(
                    f"idx: {j}",
                    f"audio RMS: {torch.sqrt(torch.mean(audio[j] ** 2))}",
                    f"parameter values: {params[j]}\n",
                )

        if i == NUM_ITERS - 1:
            break
