"""
Script to run HPO for the preset embedding framework.

Usage example:
    python src/hpo/run.py tag=<tag>

The results are exported to
`<project-root>/logs/optuna/<m_preset>_<tag>` 

Remarks:
 - If a study with the same name already exists, it will be resumed.
 - Pressing Ctrl+Z will aborted the study at the end of the current trial.

The config is defined in `configs/hpo/hpo.yaml`.
"""

# TODO: refactor script and config to allow HPO of models
from functools import partial
from pathlib import Path
import signal

import hydra
import numpy as np
import lightning as L
from lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from data.datasets import SynthDatasetPkl
from hpo.lit_module import PresetEmbeddingHPO
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


# Define a signal handler function for SIGSTP
def sigstp_handler(study: optuna.study.Study, signal, frame) -> None:
    """Handler for SIGSTP signal, aborts the current optuna study."""
    # Abort the study
    log.info("Ctrl+Z detected, the study will be aborted at the end of the current trial...")
    study.stop()


# Function to register signal handler with additional arguments
def register_signal_handler_with_args(signal_num, handler_func, *args):
    """Register a signal handler with additional arguments."""
    signal.signal(signal_num, lambda signal, frame: handler_func(*args, signal, frame))


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    """Objective function for optuna HPO."""
    # set RNG seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    trial.set_user_attr("seed", cfg.seed)

    # Sample hyperparameter values if required:
    hps = {}
    for k, v in cfg.search_space.items():
        hps[k] = getattr(trial, "suggest_" + v.type)(**v.kwargs) if isinstance(v, DictConfig) else v

    # instantiate train Dataset & DataLoader
    train_dataset = SynthDatasetPkl(cfg.path_to_train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    trial.set_user_attr("dataset/train", train_dataset.path_to_dataset.stem)

    # instantiate validation Dataset & DataLoader
    val_dataset = SynthDatasetPkl(cfg.path_to_val_dataset, mmap=False)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.num_ranks_mrr, num_workers=cfg.num_workers, shuffle=False
    )
    trial.set_user_attr("dataset/val", val_dataset.path_to_dataset.stem)

    # instantiate PresetHelper
    preset_helper = PresetHelper(
        synth_name=train_dataset.synth_name,
        params_to_exclude_str=train_dataset.configs_dict["params_to_exclude"],
    )

    # instantiate preset encoder
    preset_encoder: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg,
        out_features=train_dataset.embedding_dim,
        preset_helper=preset_helper,
        num_blocks=hps["num_blocks"],
        hidden_features=hps["hidden_features"],
    )
    trial.set_user_attr("m_preset/num_params", preset_encoder.num_parameters)
    trial.set_user_attr("m_preset/name", cfg.m_preset.name)

    # instantiate optimizer, lr_scheduler, and scheduler_config
    beta1 = hps.get("beta1", 0.9)
    beta2 = hps.get("beta2", 0.999)
    eps = hps.get("eps", 1e-8)
    optimizer = partial(Adam, betas=(beta1, beta2), eps=eps)

    lr_scheduler = (
        hydra.utils.instantiate(
            cfg.lr_scheduler,
            total_steps=np.ceil(len(train_dataset) / cfg.batch_size),
            milestone=hps["milestones"],
            final_lr=hps["learning_rate_min"],
        )
        if cfg.get("lr_scheduler")
        else None
    )
    scheduler_config = (
        OmegaConf.to_container(cfg.scheduler_config, resolve=True) if cfg.get("scheduler_config") else None
    )

    # instantiate Lightning Module
    model = PresetEmbeddingHPO(
        preset_encoder=preset_encoder,
        loss=nn.L1Loss(),
        optimizer=optimizer,
        lr=hps["learning_rate"],
        scheduler=lr_scheduler,
        scheduler_config=scheduler_config,
    )

    # instantiate logger
    logger = hydra.utils.instantiate(cfg.wandb, name=f"trial_{trial.number}") if cfg.get("wandb") else []

    # instantiate Lightning Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=cfg.metric_to_optimize)],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
    )

    # Train for one epoch
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # get metric to optimize's value (e.g., MRR score)
    metric_value = trainer.callback_metrics[cfg.metric_to_optimize].item()

    if logger:
        # log hyperparameters, seed, and dataset refs
        # log sampler and pruner config
        sp_dict = {"sampler": {}, "pruner": {}}
        for name, d in sp_dict.items():
            tmp_dict = OmegaConf.to_container(cfg[name], resolve=True)
            d["name"] = tmp_dict.get("name")
            for param, value in tmp_dict["cfg"].items():
                if param != "_target_":
                    d[f"{param}"] = value
        trainer.logger.log_hyperparams({"hps": hps, "misc": trial.user_attrs, **sp_dict})
        # terminate wandb run
        wandb.finish()

    return metric_value


@hydra.main(version_base="1.3", config_path="../../configs/hpo", config_name="hpo.yaml")
def hpo(cfg: DictConfig) -> None:
    """Main function for HPO. Config is defined in `configs/hpo/hpo.yaml`."""
    # Add stream handler of stdout to show the messages
    optuna.logging.enable_propagation()  # send messages to the root logger

    study_name = cfg.study_name  # Unique identifier of the study.

    if (Path.cwd() / f"{study_name}.db").exists():
        log.info(f"Study {study_name} already exists, resuming...")
    else:
        log.info(f"Starting new study with name {study_name}")

    storage_name = f"sqlite:///{study_name}.db"

    # load sammpler if exists else create
    if (Path.cwd() / "optuna_sampler.pkl").exists():
        log.info(f"Loading sampler from {Path.cwd() / 'optuna_sampler.pkl'}")
        sampler = torch.load(Path.cwd() / "optuna_sampler.pkl")
    else:
        sampler = hydra.utils.instantiate(cfg.sampler.cfg)

    pruner = hydra.utils.instantiate(cfg.pruner.cfg)

    study = optuna.create_study(
        study_name=str(study_name),
        storage=str(storage_name),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction=cfg.direction,
    )

    # Register the signal handler for SIGSTP with additional study argument
    register_signal_handler_with_args(signal.SIGTSTP, sigstp_handler, study)

    # Start HPO
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.num_trials - len(study.trials),
        timeout=cfg.timeout,
    )

    log.info(f"Number of finished trials: {len(study.trials)}")

    log.info("Best trial:")
    trial = study.best_trial
    log.info(f"  Value: {trial.value}")

    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info(f"    {key}: {value}")

    # Save the sampler to be loaded later if needed.
    log.info("Saving sampler...")
    with open("optuna_sampler.pkl", "wb") as f:
        torch.save(study.sampler, f)


if __name__ == "__main__":
    # import sys

    # args = ["src/hpo/run.py"]

    # sys.argv = args

    # gettrace = getattr(sys, "gettrace", None)
    # if gettrace():
    #     sys.argv = args

    hpo()  # pylint: disable=no-value-for-parameter
