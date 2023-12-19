# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
"""
Training script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
"""
import os
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl
from utils.logging import RankedLogger, log_hyperparameters
from utils.misc import get_metric_value
from utils.instantiators import instantiate_callbacks, instantiate_loggers, check_val_dataset

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)

# TODO: log stuff related to validation dataset and return appropriate metrics
# val_dataset_path to be overwriten un experiment config files


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

    log.info(f"Instantiating pre-trained audio feature extractor model <{cfg.m_audio.cfg._target_}>")
    audio_fe: nn.Module = hydra.utils.instantiate(cfg.m_audio.cfg)

    log.info(f"Instantiating training Dataset <{cfg.dataset_train.cfg._target_}>")
    dataset_train: Dataset = hydra.utils.instantiate(cfg.dataset_train.cfg, sample_rate=audio_fe.sample_rate)

    if cfg.get("dataset_val"):
        log.info("Instantiating validation Dataset & DataLoader")
        dataset_val = SynthDatasetPkl(cfg.dataset_val.path)
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg.dataset_val.loader.num_ranks,
            num_workers=cfg.dataset_val.loader.num_workers,
            shuffle=cfg.dataset_val.loader.shuffle,
            pin_memory=cfg.dataset_val.loader.pin_memory,
            drop_last=cfg.dataset_val.loader.drop_last,
        )
        check_val_dataset(cfg, dataset_train, dataset_val)
    else:
        dataset_val = None
        val_loader = None

    # intializing custom Sampler in case of one-epoch training and
    # resuming training by providing the last sample index.
    # This step is done here as it doesn't work when done in the Lightning Module.
    if cfg.dataloader_train.get("sampler"):
        log.info(f"Instantiating Sampler <{cfg.dataloader_train.sampler._target_}> and training DataLoader")
        if cfg.get("ckpt_path"):
            sampler_idx_new = torch.load(cfg.ckpt_path)["sampler_idx_last"] + 1
            log.info(f"Resuming one-epoch training from sample: {sampler_idx_new}")
        else:
            sampler_idx_new = 0
        sampler = hydra.utils.instantiate(
            cfg.dataloader_train.sampler,
            data_source=dataset_train,
            start_idx=sampler_idx_new,
        )
    else:
        log.info("No custom Sampler found, skipping...")
        sampler = None

    train_loader: DataLoader = DataLoader(
        dataset=dataset_train,
        batch_size=cfg.dataloader_train.batch_size,
        num_workers=cfg.dataloader_train.num_workers,
        pin_memory=cfg.dataloader_train.pin_memory,
        drop_last=cfg.dataloader_train.drop_last,
        sampler=sampler,
    )

    log.info(f"Instantiating preset encoder <{cfg.m_preset.cfg._target_}>")
    preset_encoder: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg, preset_helper=dataset_train.preset_helper, out_features=audio_fe.out_features
    )

    log.info(f"Instantiating Lightning Module <{cfg.solver._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.solver, audio_feature_extractor=audio_fe, preset_encoder=preset_encoder
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] | None = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "dataset_train": dataset_train,
        "dataset_val": dataset_val,
        "audio_fe": audio_fe,
        "preset_encoder": preset_encoder,
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
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.

    Args
    - `cfg`: DictConfig configuration composed by Hydra.

    Returns
    - Optional[float] with optimized metric value.
    """

    # # apply extra utilities
    # # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    # train the model
    metrics_dict, _ = train(cfg)

    # TODO: check when doing hparams optim (check get_metric_value from Lightning-hydra template)
    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    metric_value = get_metric_value(metric_dict=metrics_dict, metric_name="val/mrr")

    # return optimized metric value
    return metric_value


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
