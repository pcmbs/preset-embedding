from typing import Optional
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import wandb

from utils.misc import get_metric_value, instantiate_callbacks, instantiate_loggers
from utils.logging import RankedLogger, log_hyperparameters

# logger for this file
log = logging.getLogger(__name__)


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
        # L.seed_everything(cfg.seed, workers=True)
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating pre-trained audio feature extractor model <{cfg.m_audio.cfg._target_}>")
    audio_fe: nn.Module = hydra.utils.instantiate(cfg.m_audio.cfg)

    log.info(f"Instantiating Dataset <{cfg.dataset.cfg._target_}> and DataLoader")
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset.cfg, sample_rate=audio_fe.sample_rate)
    if cfg.dataloader.get("sampler"):
        sampler = hydra.utils.instantiate(cfg.dataloader.sampler, data_source=dataset)
    else:
        sampler = None

    train_loader: DataLoader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
        sampler=sampler,
    )

    log.info(f"Instantiating preset encoder <{cfg.m_preset.cfg._target_}>")
    preset_encoder: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg, in_features=dataset.num_used_parameters, out_features=audio_fe.out_features
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
        "dataset": dataset,
        "audio_fe": audio_fe,
        "preset_encoder": preset_encoder,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=cfg.get("ckpt_path"))

    # get metrics available to callbacks. This includes metrics logged via log().
    train_metrics = trainer.callback_metrics

    if logger:
        if cfg.logger.get("wandb"):
            # additional save the hydra config
            # under <project_name>/Runs/<run_id>/Files/.hydra if using wandb a logger
            wandb.save(
                glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
                base_path=cfg["paths"].get("output_dir"),
            )
            wandb.finish()

    return train_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.

    Args
    - `cfg`: DictConfig configuration composed by Hydra.

    Returns
    - Optional[float] with optimized metric value.
    """
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    # # apply extra utilities
    # # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # TODO: check when doing hparams optim (check get_metric_value from Lightning-hydra template)
    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name="train/loss")

    # return optimized metric value
    return metric_value


if __name__ == "__main__":
    args = [
        "src/div_check/train_debug.py",
        "experiment=exp-iter2",
        "trainer.max_steps=20",
        "logger=csv",
        "ckpt_path=logs/mlp_relu_fnorm-mn04/tal_noisemaker-iter/dev_2023-11-22_09-46-14/checkpoints/mlp_relu_fnorm_mn04_tal_noisemaker_s10.ckpt",
    ]

    # sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    main()  # pylint: disable=no-value-for-parameter
