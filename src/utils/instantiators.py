# pylint: disable=W0212,W1203
"""
Helper functions for instantiating lightning callbacks from DictConfig objects.

Source: https://github.com/ashleve/lightning-hydra-template
"""

from typing import List, Tuple

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.info("No callback configs found. Skipping...")
        return None

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def check_val_dataset(cfg: DictConfig, dataset_train: Dataset, dataset_val: Dataset) -> None:
    assert (
        dataset_val.audio_fe_name == cfg["m_audio"]["name"]
    ), f"the audio model used for training and validation should be the same: {cfg['m_audio']['name']} != {dataset_val.audio_fe_name}"
    assert (
        dataset_val.synth_name == dataset_train.synth_name
    ), f"the synthesizer used for training and validation should be the same: {dataset_train.synth_name} != {dataset_val.synth_name}"
    assert (
        dataset_val.configs_dict["params_to_exclude"] == dataset_train.preset_helper.excl_params_str
    ), "the params_to_exclude used for training and validation should be the same."
    assert dataset_val.num_used_synth_params == dataset_train.num_used_parameters
    for attr in ["render_duration_in_sec", "midi_note", "midi_velocity", "midi_duration_in_sec"]:
        assert dataset_val.configs_dict[attr] == getattr(
            dataset_train, attr
        ), f"the {attr} used for training and validation should be the same: {getattr(dataset_train, attr)} != {dataset_val.configs_dict[attr]}"
