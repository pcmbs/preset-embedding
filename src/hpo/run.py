from functools import partial

import hydra
import lightning as L
from lightning import Trainer
from omegaconf import ListConfig, OmegaConf, DictConfig
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from data.datasets import SynthDatasetPkl
from hpo.lit_module import PresetEmbeddingHp
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    ss = cfg.search_space

    # set RNG seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    trial.set_user_attr("seed", cfg.seed)

    # Sample hyperparameter values if required:

    # if isinstance(ss.train_batch_size, list):
    #     train_batch_size = trial.suggest_categorical("train_batch_size", ss.train_batch_size)
    # else:
    #     train_batch_size = ss.train_batch_size

    if isinstance(ss.num_blocks, ListConfig):
        num_blocks = trial.suggest_int("num_blocks", *ss.num_blocks)
    else:
        num_blocks = ss.num_blocks
    trial.set_user_attr("m_preset/num_blocks", num_blocks)

    if isinstance(ss.hidden_features, ListConfig):
        hidden_features = trial.suggest_int("hidden_features", *ss.hidden_features, log=True)
    else:
        hidden_features = ss.hidden_features
    trial.set_user_attr("m_preset/hidden_features", hidden_features)

    if isinstance(ss.optimizer_names, ListConfig):
        optimizer_name = trial.suggest_categorical("optimizer", ss.optimizer_names)
    else:
        optimizer_name = ss.optimizer_names
    trial.set_user_attr("solver/optimizer", optimizer_name)

    if isinstance(ss.learning_rate, ListConfig):
        learning_rate = trial.suggest_float("lr", *ss.learning_rate, log=True)
    else:
        learning_rate = ss.learning_rate
    trial.set_user_attr("solver/lr", learning_rate)

    # instantiate train Dataset & DataLoader
    train_dataset = SynthDatasetPkl(cfg.path_to_train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    trial.set_user_attr("dataset/train", train_dataset.path_to_dataset.stem)
    trial.set_user_attr("solver/batch_size", cfg.batch_size)

    # instantiate validation Dataset & DataLoader
    val_dataset = SynthDatasetPkl(cfg.path_to_val_dataset)
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
        num_blocks=num_blocks,
        hidden_features=hidden_features,
    )
    trial.set_user_attr("m_preset/num_params", preset_encoder.num_parameters)
    trial.set_user_attr("m_preset/name", cfg.m_preset.name)
    # # instantiate optimizer
    optimizer = partial(getattr(torch.optim, optimizer_name), lr=learning_rate)

    # has_lr_warmup = trial.suggest_categorical("lr_warmup", [True, False])
    # if has_lr_warmup:
    #     lr_warmup_total_iters = trial.suggest_int("lr_warmup_total_iters", *ss.lr_warmup_total_iters)
    #     lr_warmup_start_factor = trial.suggest_float("lr_warmup_start_factor", *ss.lr_warmup_start_factor)
    #     lr_scheduler = partial(
    #         torch.optim.lr_scheduler.LinearLR,
    #         start_factor=lr_warmup_start_factor,
    #         total_iters=lr_warmup_total_iters,
    #     )
    # else:
    #     lr_scheduler = None
    lr_scheduler = None
    scheduler_config = (
        OmegaConf.to_container(cfg.scheduler_config, resolve=True) if cfg.get("scheduler_config") else None
    )
    # instantiate Lightning Module
    model = PresetEmbeddingHp(
        preset_encoder=preset_encoder,
        loss=nn.L1Loss(),
        optimizer=optimizer,
        lr=learning_rate,
        scheduler=lr_scheduler,
        scheduler_config=scheduler_config,
    )

    # instantiate logger
    logger = hydra.utils.instantiate(cfg.wandb) if cfg.get("wandb") else []

    # instantiate Lightning Trainer
    trainer = Trainer(
        max_epochs=1,
        check_val_every_n_epoch=1,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=cfg.metric_to_optimize)],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    # Train for one epoch
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # get metric to optimize's value (e.g., MRR score)
    metric_value = trainer.callback_metrics[cfg.metric_to_optimize].item()

    if logger:
        # log hyperparameters, seed, and dataset refs
        trainer.logger.log_hyperparams(trial.user_attrs)
        # log sampler and pruner config
        sampler_pruner_dict = {}
        for i in ["sampler", "pruner"]:
            tmp_dict = OmegaConf.to_container(cfg[i], resolve=True)
            sampler_pruner_dict["sampler/name"] = tmp_dict.get("name")
            for k, v in tmp_dict["cfg"].items():
                if k != "_target_":
                    sampler_pruner_dict[f"sampler/{k}"] = v
        trainer.logger.log_hyperparams(sampler_pruner_dict)
        # terminate wandb run
        wandb.finish()

    return metric_value


@hydra.main(version_base="1.3", config_path="../../configs/hpo", config_name="hpo.yaml")
def hpo(cfg: DictConfig) -> None:
    # Add stream handler of stdout to show the messages
    optuna.logging.enable_propagation()  # send messages to the root logger

    # # Create a new study and start optimizing
    study_name = cfg.study_name  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"

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
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.num_trials)

    log.info(f"Number of finished trials: {len(study.trials)}")

    log.info("Best trial:")
    trial = study.best_trial
    log.info(f"  Value: {trial.value}")

    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info(f"    {key}: {value}")

    # TODO: maybe add possibility to resume study
    # Save the sampler with pickle to be loaded later.
    # with open("sampler.pkl", "wb") as f:
    #     pickle.dump(study.sampler, f)

    # ######### Resume Study (with sampler)
    # restored_sampler = pickle.load(open("sampler.pkl", "rb"))
    # study = optuna.create_study(
    #     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
    # )
    # study.optimize(objective, n_trials=3)


if __name__ == "__main__":
    hpo()  # pylint: disable=no-value-for-parameter
