import hydra
from omegaconf import DictConfig
import lightning as L
from lightning import LightningModule, Trainer
from lightning.pytorch.tuner import Tuner
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from data.datasets import SynthDatasetPkl
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


def _single_run(cfg: DictConfig, run_id: int) -> None:
    L.seed_everything(cfg.start_seed + run_id)

    # instantiate Dataset & DataLoader
    dataset = SynthDatasetPkl(cfg.path_to_dataset)
    dataloader = DataLoader(dataset, **cfg.loader)

    # instantiate PresetHelper
    preset_helper = PresetHelper(
        synth_name=dataset.synth_name, params_to_exclude_str=dataset.configs_dict["params_to_exclude"]
    )

    # instantiate preset encoder
    preset_encoder: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg, preset_helper=preset_helper, out_features=dataset.embedding_dim
    )

    # instantiate optimizer
    optimizer: Optimizer = hydra.utils.instantiate(cfg.optimizer.cfg)

    # instantiate Lightning Module
    model: LightningModule = hydra.utils.instantiate(
        cfg.solver, preset_encoder=preset_encoder, optimizer=optimizer, lr=1e-3
    )

    # instantiate Lightning Trainer
    trainer = Trainer(
        max_epochs=-1,
        check_val_every_n_epoch=None,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    # instantiate Tuner
    tuner = Tuner(trainer)

    # # Run learning rate finder
    lr_finder_ = tuner.lr_find(
        model=model,
        train_dataloaders=dataloader,
        method="fit",
        **cfg.lr_finder,
    )

    # get suggestion
    suggested_lr = lr_finder_.suggestion()
    log.info(f"Suggested LR: {suggested_lr}")

    results_dict = lr_finder_.results
    results_dict["suggested_lr"] = suggested_lr

    return results_dict


@hydra.main(version_base="1.3", config_path="../../configs/lr_finder", config_name="lr_finder.yaml")
def lr_finder(cfg: DictConfig) -> None:
    """
    Docstring TODO
    """
    results = []
    for run_id in range(cfg.num_run):
        results.append(_single_run(cfg, run_id))

    # store results as pkl file
    with open("lr_finder_results.pkl", "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    lr_finder()  # pylint: disable=no-value-for-parameter
