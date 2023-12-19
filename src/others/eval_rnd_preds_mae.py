# pylint: disable=W1203
"""
Script used to compute the value of the MAE of the random predictions for a given audio model and synthesizer.
"""

# TODO: refactor

import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from tqdm import tqdm

from data.datasets import SynthDataset
from utils.synth import PresetHelper
from models import audio as audio_models

log = logging.getLogger(__name__)

PATH_TO_STATS = (
    Path(os.environ["PROJECT_ROOT"])
    / "data"
    / "eval"
    / "stats_for_rnd_predictions"
    / "mn04_NoiseMakerDataset_65536_2023-11-30_15-14-41"
)

# Load configs.pkl
with open(PATH_TO_STATS / "configs.pkl", "rb") as f:
    configs_dict = torch.load(f)

configs_dict.pop("device")
configs_dict.pop("num_samples")
configs_dict.pop("batch_size")
configs_dict.pop("num_iters")
configs_dict.pop("seed_offset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 8_192  # number of samples from which to compute the MAE
BATCH_SIZE = 64
NUM_ITERS = int(np.floor(NUM_SAMPLES / BATCH_SIZE))

RND_PRED_SEED = 42

SYNTH = configs_dict["synth"]
PARAMETERS_TO_EXCLUDE_STR = configs_dict["params_to_exclude"]
RENDER_DURATION_IN_SEC = configs_dict["render_duration_in_sec"]
MIDI_NOTE = configs_dict["midi_note"]
MIDI_VELOCITY = configs_dict["midi_velocity"]
MIDI_DURATION_IN_SEC = configs_dict["midi_duration_in_sec"]
SEED_OFFSET = 101

AUDIO_FE = configs_dict["audio_fe"]

filename = PATH_TO_STATS.stem

EXPORT_PATH = Path(os.environ["PROJECT_ROOT"]) / "div_check" / "eval" / "rnd_preds_mae" / filename
if not EXPORT_PATH.exists():
    EXPORT_PATH.mkdir(parents=True)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXPORT_PATH / "MAE_rnd_preds.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

configs_dict["rnd_preds_seed"] = RND_PRED_SEED
configs_dict["num_samples"] = NUM_SAMPLES

if __name__ == "__main__":
    # log the configs dict
    for k, v in configs_dict.items():
        log.info(f"{k}: {v}")

    # export the configs dict as pickle
    with open(EXPORT_PATH / "configs.pkl", "wb") as f:
        torch.save(configs_dict, f)

    # Load mean and std
    with open(PATH_TO_STATS / "mean.pkl", "rb") as f:
        audio_repr_mean = torch.load(f)
    with open(PATH_TO_STATS / "std.pkl", "rb") as f:
        audio_repr_std = torch.load(f)

    audio_fe = getattr(audio_models, AUDIO_FE[0])()
    audio_fe.to(DEVICE)
    audio_fe.eval()

    p_helper = PresetHelper(synth_name=SYNTH, params_to_exclude_str=PARAMETERS_TO_EXCLUDE_STR)

    dataset = SynthDataset(
        preset_helper=p_helper,
        dataset_size=NUM_SAMPLES,
        seed_offset=SEED_OFFSET,
        sample_rate=audio_fe.sample_rate,
        render_duration_in_sec=RENDER_DURATION_IN_SEC,
        midi_note=MIDI_NOTE,
        midi_velocity=MIDI_VELOCITY,
        midi_duration_in_sec=MIDI_DURATION_IN_SEC,
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(RND_PRED_SEED)

    pbar = tqdm(dataloader, total=NUM_ITERS, dynamic_ncols=True)

    loss = L1Loss()

    loss_per_batch = []

    for i, (_, audio, _) in enumerate(pbar):
        with torch.no_grad():
            audio = audio.to(DEVICE)
            audio_repr = audio_fe(audio)
            rnd_pred = torch.normal(audio_repr_mean, audio_repr_std)

        loss_per_batch.append(loss(rnd_pred, audio_repr.cpu()).item())

        if i == NUM_ITERS - 1:
            break

    log.info("")
    log.info(f"Mean MAE: {np.mean(loss_per_batch):.4f}")
    log.info(f"Std MAE: {np.std(loss_per_batch):.4f}")
