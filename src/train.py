# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
"""
Training script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
"""
import os
from typing import Any, Dict, List

import hydra
import lightning as L
from torch import nn
from torch.utils.data import DataLoader
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl
from utils.logging import RankedLogger, log_hyperparameters
from utils.instantiators import instantiate_callbacks, instantiate_loggers, check_val_dataset
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> Dict[str, Any]:
    """
    Trains the mmodel

    Args
    - `cfg`: A DictConfig configuration composed by Hydra.

    Returns
    - A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating training Dataset: {cfg.train_dataset.path}")
    train_dataset = SynthDatasetPkl(cfg.train_dataset.path)
    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_dataset.loader.batch_size,
        num_workers=cfg.train_dataset.loader.num_workers,
        pin_memory=cfg.train_dataset.loader.pin_memory,
        drop_last=cfg.train_dataset.loader.drop_last,
    )

    log.info(f"Instantiating validation Dataset: {cfg.val_dataset.path}")
    val_dataset = SynthDatasetPkl(cfg.val_dataset.path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataset.loader.num_ranks,
        num_workers=cfg.val_dataset.loader.num_workers,
        shuffle=cfg.val_dataset.loader.shuffle,
        pin_memory=cfg.val_dataset.loader.pin_memory,
        drop_last=True,
    )
    check_val_dataset(train_dataset, val_dataset)

    log.info(f"Instantiating Preset Helper for synth {train_dataset.synth_name} and excluded params:")
    log.info(f"{train_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=train_dataset.synth_name,
        parameters_to_exclude=train_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.m_preset.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg, out_features=train_dataset.embedding_dim, preset_helper=preset_helper
    )

    log.info(f"Instantiating Lightning Module <{cfg.solver._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.solver, preset_encoder=m_preset)

    log.info("Instantiating Callbacks...")
    callbacks: List[Callback] | None = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "m_preset": m_preset,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.get("ckpt_path"),
        )

    # get metrics available to callbacks. This includes metrics logged via log().
    metrics_dict = trainer.callback_metrics

    if logger:
        if cfg.logger.get("wandb"):
            # additional save the hydra config
            # under <project_name>/Runs/<run_id>/Files/.hydra if using wandb a logger
            wandb.save(
                glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
                base_path=cfg["paths"].get("output_dir"),
            )
            wandb.finish()

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs/train", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.
    """
    # train the model
    metrics_dict, _ = train(cfg)

    log.info(f"Metrics: {metrics_dict}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
