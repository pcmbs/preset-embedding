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
import wandb

from data.datasets import SynthDatasetPkl
from hpo.lit_module import PresetEmbeddingHp
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    # set RNG seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    trial.set_user_attr("seed", cfg.seed)

    # Sample hyperparameter values if required:
    hps = {}
    for k, v in cfg.search_space.items():
        if isinstance(v, DictConfig):
            hps[k] = getattr(trial, "suggest_" + v.type)(**v.kwargs) if isinstance(v, DictConfig) else v

    # instantiate train Dataset & DataLoader
    train_dataset = SynthDatasetPkl(cfg.path_to_train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=hps["batch_size"], shuffle=True, num_workers=cfg.num_workers
    )
    trial.set_user_attr("dataset/train", train_dataset.path_to_dataset.stem)

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
        num_blocks=hps["num_blocks"],
        hidden_features=hps["hidden_features"],
    )
    trial.set_user_attr("m_preset/num_params", preset_encoder.num_parameters)
    trial.set_user_attr("m_preset/name", cfg.m_preset.name)
    # # instantiate optimizer
    optimizer = partial(getattr(torch.optim, hps["optimizer"]))

    lr_scheduler = None
    scheduler_config = (
        OmegaConf.to_container(cfg.scheduler_config, resolve=True) if cfg.get("scheduler_config") else None
    )
    # instantiate Lightning Module
    model = PresetEmbeddingHp(
        preset_encoder=preset_encoder,
        loss=nn.L1Loss(),
        optimizer=optimizer,
        lr=hps["learning_rate"],
        scheduler=lr_scheduler,
        scheduler_config=scheduler_config,
    )

    # instantiate logger
    logger = hydra.utils.instantiate(cfg.wandb) if cfg.get("wandb") else []

    # instantiate Lightning Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=1,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
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
    import sys

    args = ["src/hpo/run.py"]

    sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    hpo()  # pylint: disable=no-value-for-parameter
