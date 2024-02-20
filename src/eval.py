# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
# pylint: disable=E1120:no-value-for-parameter
"""
Evaluation script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
"""
from typing import Any, Dict

import hydra
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl
from models.lit_module import PresetEmbeddingLitModule
from utils.logging import RankedLogger  # , log_hyperparameters
from utils.instantiators import instantiate_loggers
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """
    Evaluates the model

    Args
    - `cfg`: A DictConfig configuration composed by Hydra.

    Returns
    - A tuple wih computed metrics.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating validation Dataset: {cfg.dataset.path}")
    test_dataset = SynthDatasetPkl(cfg.dataset.path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.loader.num_ranks,
        num_workers=cfg.dataset.loader.num_workers,
        shuffle=cfg.dataset.loader.shuffle,
        pin_memory=cfg.dataset.loader.pin_memory,
        drop_last=cfg.dataset.loader.drop_last,
    )

    log.info(f"Instantiating Preset Helper for synth {test_dataset.synth_name} and excluded params:")
    log.info(f"{test_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=test_dataset.synth_name,
        params_to_exclude_str=test_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.model.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.model.cfg, out_features=test_dataset.embedding_dim, preset_helper=preset_helper
    )

    log.info(f"Loading checkpoint from {cfg.ckpt_path}...")
    model = PresetEmbeddingLitModule.load_from_checkpoint(cfg.ckpt_path, preset_encoder=m_preset)
    model.freeze()

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # object_dict = {
    #     "cfg": cfg,
    #     "test_dataset": test_dataset,
    #     "m_preset": m_preset,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters...")
    #     log_hyperparameters(object_dict)

    log.info("Starting evaluation...")

    # get metrics available to callbacks. This includes metrics logged via log().
    metrics_dict = 0  # trainer.callback_metrics

    # if logger:
    #     if cfg.logger.get("wandb"):
    #         # additional save the hydra config
    #         # under <project_name>/Runs/<run_id>/Files/.hydra if using wandb a logger
    #         wandb.save(
    #             glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
    #             base_path=cfg["paths"].get("output_dir"),
    #         )
    #         wandb.finish()

    return metrics_dict


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.
    """
    # train the model
    metrics_dict = evaluate(cfg)

    log.info(f"Metrics: {metrics_dict}")


if __name__ == "__main__":
    import sys

    args = ["src/eval.py", "model=mlp_relu_oh_talnm", "task_name=debug"]

    sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    main()  # pylint: disable=no-value-for-parameter
