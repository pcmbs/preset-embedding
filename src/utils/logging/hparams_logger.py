"""
Module implementing a function to log hyperparameters.

Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/logging_utils.py 
"""

from typing import Any, Dict
import os

from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
import wandb

from .ranked_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Args
    - `object_dict`: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"dataset_train"`: The torch dataset used for training.
        - `"dataset_val"`: The torch dataset used for validation.
        - `"audio_fe"`: nn.Module for the audio feature extractor model.
        - `"preset_encoder"`: nn.Module for the preset encoder model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # general hyperparameters
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # training dataset related hyperparameters
    dataset = object_dict["dataset_train"]
    for k, v in cfg["dataset_train"]["cfg"].items():
        if k not in ["path_to_plugin", "_target_", "preset_helper"]:
            hparams[f"dataset_train/{k}"] = v
    hparams["dataset_train/name"] = cfg["dataset_train"]["cfg"]["_target_"].split(".")[-1]
    hparams["dataset_train/synth_name"] = dataset.synth_name
    hparams["dataset_train/num_used_synth_params"] = dataset.num_used_parameters
    hparams["dataset_train/excl_synth_params"] = cfg["dataset_train"]["cfg"]["preset_helper"][
        "params_to_exclude_str"
    ]

    # validation dataset related hyperparameters
    if object_dict["dataset_val"]:
        dataset_val = object_dict["dataset_val"]
        for k, v in dataset_val.configs_dict.items():
            hparams[f"dataset_val/{k}"] = v
        hparams["dataset_val/num_used_synth_params"] = dataset_val.num_used_synth_params
        hparams["dataset_val/num_ranks"] = cfg["dataset_val"]["loader"]["num_ranks"]
        hparams["dataset_val/num_samples_per_rank"] = int(
            len(dataset_val) // cfg["dataset_val"]["loader"]["num_ranks"]
        )

    # pre-trained audio model related hyperparameters
    audio = object_dict["audio_fe"]
    hparams["m_audio/name"] = cfg["m_audio"]["name"]
    hparams["m_audio/num_params"] = audio.num_parameters
    hparams["m_audio/out_features"] = audio.out_features
    hparams["m_audio/includes_mel"] = audio.includes_mel

    # preset encoder related hyperparameters
    preset = object_dict["preset_encoder"]
    for k, v in cfg["m_preset"]["cfg"].items():
        if k == "_target_":
            hparams["m_preset/name"] = v.split(".")[-1]
        elif k in ["embedding_kwargs", "block_kwargs"]:
            for kk, vv in v.items():
                hparams[f"m_preset/{k}/{kk}"] = vv
        else:
            hparams[f"m_preset/{k}"] = v
    hparams["m_preset/num_params"] = preset.num_parameters

    # solver related hyperparameters
    hparams["solver/loss"] = cfg["solver"]["loss"]["_target_"].split(".")[-1]

    hparams["solver/optim/name"] = cfg["solver"]["optimizer"]["_target_"].split(".")[-1]
    for k, v in cfg["solver"]["optimizer"].items():
        if k not in ["_target_", "_partial_"]:
            hparams[f"solver/optim/{k}"] = v

    if cfg["solver"].get("scheduler"):
        hparams["solver/sched/name"] = cfg["solver"]["scheduler"]["_target_"].split(".")[-1]
        for k, v in cfg["solver"]["scheduler"].items():
            if k not in ["_target_", "_partial_"]:
                hparams[f"solver/sched/{k}"] = v

    if cfg["solver"].get("scheduler_config"):
        for k, v in cfg["solver"]["scheduler_config"].items():
            if k != "monitor":
                hparams[f"solver/sched/{k}"] = v

    # training related hyperparameters
    hparams["dataloader_train"] = cfg["dataloader_train"]
    for k, v in cfg["trainer"].items():
        if (k not in ["_target_", "default_root_dir"]) and (v is not None):
            hparams[f"trainer/{k}"] = v

    # send hparams to logger
    trainer.logger.log_hyperparams(hparams)
    # additional save the hydra config if using wandb a logger
    # (later in src/train.py since output_dir doesn't exist yet).
