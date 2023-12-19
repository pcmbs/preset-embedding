# pylint: disable=W1203
"""
Script to compute the mean and standard deviation across the number of output features of the embeddings for a given audio model.
The resulting log file, mean and std are exported to
`<project-root>/data/eval/stats_for_rnd_predictions/<audio_fe>_<dataset>_<NUM_SAMPLES>_<timestamp>/`
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

################ SYNTH SETTINGS
SYNTH = "tal_noisemaker"

PARAMETERS_TO_EXCLUDE_STR = (
    "master_volume",
    "voices",
    "lfo_1_sync",
    "lfo_1_keytrigger",
    "lfo_2_sync",
    "lfo_2_keytrigger",
    "envelope*",
    "portamento*",
    "pitchwheel*",
    "delay*",
)

################ DATASET SETTINGS
NUM_SAMPLES = 65_536
BATCH_SIZE = 64
NUM_ITERS = int(np.floor(NUM_SAMPLES / BATCH_SIZE))


RENDER_DURATION_IN_SEC = 4.0
MIDI_NOTE = 60
MIDI_VELOCITY = 110
MIDI_DURATION_IN_SEC = 2.0
SEED_OFFSET = 100

################ AUDIO MODEL SETTINGS
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

################ EXPORT & LOGGING
folder_name = f"{AUDIO_FE[0]}_{SYNTH}_{NUM_SAMPLES}_{dt_string}"

RELATIVE_PATH = Path("data") / "eval" / "stats_for_rnd_predictions" / folder_name

EXPORT_PATH = Path(os.environ["PROJECT_ROOT"]) / RELATIVE_PATH
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

configs_dict = {
    "relative_path_to_stats": RELATIVE_PATH,
    "device": DEVICE,
    "synth": SYNTH,
    "params_to_exclude": PARAMETERS_TO_EXCLUDE_STR,
    "num_samples": NUM_SAMPLES,
    "batch_size": BATCH_SIZE,
    "num_iters": NUM_ITERS,
    "render_duration_in_sec": RENDER_DURATION_IN_SEC,
    "midi_note": MIDI_NOTE,
    "midi_velocity": MIDI_VELOCITY,
    "midi_duration_in_sec": MIDI_DURATION_IN_SEC,
    "seed_offset": SEED_OFFSET,
    "audio_fe": AUDIO_FE,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(EXPORT_PATH / "configs.log")),
        logging.StreamHandler(sys.stdout),
    ],
)


if __name__ == "__main__":
    # log the configs dict
    for k, v in configs_dict.items():
        log.info(f"{k}: {v}")

    # export the configs dict as pickle
    with open(EXPORT_PATH / "configs.pkl", "wb") as f:
        torch.save(configs_dict, f)

    # compute mean and std
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
    pbar = tqdm(dataloader, total=NUM_ITERS, dynamic_ncols=True)
    audio_repr = []

    for i, (_, audio, _) in enumerate(pbar):
        with torch.no_grad():
            audio = audio.to(DEVICE)

        audio_repr.append(audio_fe(audio))

        if i == NUM_ITERS - 1:
            break

    audio_repr = torch.cat(audio_repr, dim=0)
    audio_repr_mean = audio_repr.mean(dim=0)
    audio_repr_std = audio_repr.std(dim=0)

    # export mean and std to pickle
    with open(EXPORT_PATH / "mean.pkl", "wb") as f:
        torch.save(audio_repr_mean.cpu(), f)
    with open(EXPORT_PATH / "std.pkl", "wb") as f:
        torch.save(audio_repr_std.cpu(), f)
