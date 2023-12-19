# pylint: disable=W1203
"""
Script to check computing time with different number of workers, pinned memory (with and without non-blocking) 
"""

# TODO: refactor

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from timeit import default_timer

import torch
from torch.utils.data import DataLoader
from data.noisemaker_iter_dataset import NoiseMakerIterDataset


dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log = logging.getLogger(__name__)

LOG_PATH = Path(os.environ["PROJECT_ROOT"]) / "div_check" / "profiler"
if not LOG_PATH.exists():
    LOG_PATH.mkdir(parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH / f"prof_num_workers_{dt_string}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_ITERS = 128

cfg_dict = {
    "batch_size": BATCH_SIZE,
    "num_iters": NUM_ITERS,
    "device": DEVICE,
}

if __name__ == "__main__":
    for k, v in cfg_dict.items():
        log.info(f"{k}: {v}")

    dataset = NoiseMakerIterDataset(
        sample_rate=44_100,
        render_duration_in_sec=4.0,
        midi_note=60,
        midi_velocity=100,
        midi_duration_in_sec=1.0,
        base_seed=456,
    )

    for w in [0, 2, 4, 8, 16, 32, 64]:
        for p in [False, True]:
            for n in [False, True]:
                log.info(f"Run settings: num_workers={w}, pin_memory={p}, non_blocking={n}")
                train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=w, pin_memory=p)

                start = default_timer()
                for i, (midi, params, audio) in enumerate(train_loader):
                    if i == NUM_ITERS:
                        break
                    midi = midi.to(DEVICE, non_blocking=n)
                    params = params.to(DEVICE, non_blocking=n)
                    audio = audio.to(DEVICE, non_blocking=n)

                log.info(f"    Execution time: {default_timer() - start:.2f}s")

    # num workers check on local machine (num_iters=100, batch_size=32)
    # 8 workers: for 100 iterations -> good trade-off (5x faster)
    #   batch_size=32: 65.85s (loading on cpu)
    #   batch_size=64: 132.56s (loading on cpu)
    # 15 workers: 52.55s  -> best (6x faster)
