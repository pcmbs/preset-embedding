"""
Module used to debug/print the hydra config
"""
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# uncomment if the env variables were not set (e.g., using docker run --env-file .env)
# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env for hydra config


@hydra.main(version_base=None, config_path="../../configs/train", config_name="train")
def cfg_debug(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("breakpoint me!")


if __name__ == "__main__":
    args = ["src/div_check/cfg_debug.py", "experiment=exp-iter-val", "task_name=debug"]

    sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    cfg_debug()  # pylint: disable=E1120:no-value-for-parameter̦
