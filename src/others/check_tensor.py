from pathlib import Path
import os

import torch

PROJECT_ROOT = Path("/workspaces/preset-embedding")

PATH_TO_DATASET = (
    PROJECT_ROOT / "data" / "datasets" / "tal_noisemaker_mn04_size=2000_seed=0_pkl_interrupt_test"
)

if (PATH_TO_DATASET / "resume_state.pkl").exists():
    with open(PATH_TO_DATASET / "resume_state.pkl", "rb") as f:
        saved_data = torch.load(f)
        start_index = saved_data["start_index"]
        audio_embeddings = saved_data["audio_embeddings"]
        synth_params = saved_data["synth_params"]

else:
    with open(PATH_TO_DATASET / "audio_embeddings.pkl", "rb") as f:
        audio_embeddings = torch.load(f)
    with open(PATH_TO_DATASET / "synth_params.pkl", "rb") as f:
        synth_params = torch.load(f)


print("breakpoint me!")
