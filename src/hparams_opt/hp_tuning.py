from functools import partial

import hydra
import lightning as L
from lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.datasets import SynthDatasetPkl
from hparams_opt.lit_module import PresetEmbeddingHp
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    ss_cfg = cfg.search_space

    # set RNG seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # instantiate train Dataset & DataLoader
    train_dataset = SynthDatasetPkl(cfg.path_to_train_dataset)
    train_batch_size = trial.suggest_categorical("train_batch_size", ss_cfg.train_batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    # instantiate validation Dataset & DataLoader
    val_dataset = SynthDatasetPkl(cfg.path_to_val_dataset)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.num_ranks_mrr, num_workers=cfg.num_workers, shuffle=False
    )

    # instantiate PresetHelper
    preset_helper = PresetHelper(
        synth_name=train_dataset.synth_name,
        params_to_exclude_str=train_dataset.configs_dict["params_to_exclude"],
    )

    # instantiate preset encoder
    num_blocks = trial.suggest_int("num_blocks", *ss_cfg.num_blocks)
    hidden_features = trial.suggest_int("hidden_features", *ss_cfg.hidden_features, log=True)
    preset_encoder: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg,
        out_features=train_dataset.embedding_dim,
        preset_helper=preset_helper,
        num_blocks=num_blocks,
        hidden_features=hidden_features,
    )

    # # instantiate optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ss_cfg.optimizer_names)
    learning_rate = trial.suggest_float("lr", *ss_cfg.learning_rate, log=True)
    optimizer = partial(getattr(torch.optim, optimizer_name), lr=learning_rate)

    has_lr_warmup = trial.suggest_categorical("lr_warmup", [True, False])
    if has_lr_warmup:
        lr_warmup_total_iters = trial.suggest_int("lr_warmup_total_iters", *ss_cfg.lr_warmup_total_iters)
        lr_warmup_start_factor = trial.suggest_float("lr_warmup_start_factor", *ss_cfg.lr_warmup_start_factor)
        lr_scheduler = partial(
            torch.optim.lr_scheduler.LinearLR,
            start_factor=lr_warmup_start_factor,
            total_iters=lr_warmup_total_iters,
        )
    else:
        lr_scheduler = None

    # instantiate Lightning Module
    model = PresetEmbeddingHp(
        preset_encoder=preset_encoder,
        loss=nn.L1Loss(),
        optimizer=optimizer,
        lr=learning_rate,
        scheduler=lr_scheduler,
        scheduler_config=OmegaConf.to_container(cfg.scheduler_config, resolve=True),
    )

    # instantiate Lightning Trainer
    trainer = Trainer(
        max_epochs=1,
        check_val_every_n_epoch=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/mrr")],
    )

    # Train for one epoch
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # get validation metric (MRR score)
    mrr_score = trainer.callback_metrics["val/mrr"].item()

    return mrr_score


@hydra.main(version_base="1.3", config_path="../../configs/hparams_opt", config_name="optuna.yaml")
def hp_tuning(cfg: DictConfig) -> None:
    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

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
    hp_tuning()  # pylint: disable=no-value-for-parameter
