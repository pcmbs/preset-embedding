"""
Script used to analyze a checkpoint file.
"""
import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

M_AUDIO = "mn04"
M_PRESET = "mlp_relu_fnorm"
DATASET = "tal_noisemaker"
TR_TYPE = "iter"

DAY = "2023-11-21"


def get_ckpt_path(run_timestamp: str, identifier: str) -> Path:
    return (
        PROJECT_ROOT
        / "logs"
        / f"{M_PRESET}-{M_AUDIO}"
        / f"{DATASET}-{TR_TYPE}"
        / f"dev_{DAY}_{run_timestamp}"
        / "checkpoints"
        / f"{M_PRESET}_{M_AUDIO}_{DATASET}_{identifier}.ckpt"
    )


CKPT_PATH_1 = get_ckpt_path("17-26-07", "s50")

CKPT_PATH_2 = get_ckpt_path("17-41-24", "s100")

# CKPT_PATH = PROJECT_ROOT / "checkpoints" / "mn04_as_mAP_432.pt"

if __name__ == "__main__":
    # ckpt_1 = torch.load(CKPT_PATH_1)
    # ckpt_2 = torch.load(CKPT_PATH_2)
    ckpt_2 = torch.load(PROJECT_ROOT / "div_check" / "test.ckpt")

    # for key in ckpt_1["state_dict"].keys():
    #     print(
    #         f"max abs error for {key}:\n   {(ckpt_1['state_dict'][key]-ckpt_2['state_dict'][key]).abs().max()}"
    #     )
    print("")
