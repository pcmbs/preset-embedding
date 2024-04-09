# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
# pylint: disable=E1120:no-value-for-parameter
"""
Evaluation script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

Usage example:
    python src/eval.py model=<model-name>

See configs/eval/model for available models
"""
from pathlib import Path
import pickle

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from torch import nn
from omegaconf import DictConfig
import torch
import wandb

from data.datasets import SynthDatasetPkl
from models.lit_module import PresetEmbeddingLitModule
from utils.evaluation import one_vs_all_eval, non_overlapping_eval, eval_logger
from utils.logging import RankedLogger
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="eval.yaml")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluation pipeline for the preset embedding framework.

    Args
    - `cfg`: A DictConfig configuration composed by Hydra.

    Return
    TODO
    """
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    L.seed_everything(cfg.seed)

    ##### Initializing datasets, models, and logger
    log.info(f"Instantiating hand-crafted presets Dataset: {cfg.synth.dataset_path}")
    hc_dataset = SynthDatasetPkl(cfg.synth.dataset_path)

    log.info(f"Instantiating random presets Dataset: {cfg.synth.rnd_dataset_path}")
    rnd_dataset = SynthDatasetPkl(cfg.synth.rnd_dataset_path)

    assert hc_dataset.configs_dict["params_to_exclude"] == rnd_dataset.configs_dict["params_to_exclude"]

    log.info(f"Instantiating Preset Helper for synth {hc_dataset.synth_name} and excluded params:")
    log.info(f"{hc_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=hc_dataset.synth_name,
        parameters_to_exclude=hc_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.model.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.model.cfg, out_features=hc_dataset.embedding_dim, preset_helper=preset_helper
    )

    if cfg.get("wandb"):
        log.info("Instantiating wandb logger...")
        run = wandb.init(**cfg.wandb)

    log.info(f"Loading checkpoint from {cfg.ckpt_path}...")
    model = PresetEmbeddingLitModule.load_from_checkpoint(cfg.ckpt_path, preset_encoder=m_preset)
    model.to(device)
    model.freeze()

    ##### Computing Evaluation Metrics
    num_hc_presets = cfg.subset_size if 0 < cfg.subset_size < len(hc_dataset) else len(hc_dataset)
    log.info(f"Computing evaluation metrics on {num_hc_presets} hand-crafted presets (one-vs-all)...")
    hc_mrr, hc_top_k_mrr, hc_ranks_dict, hc_loss = one_vs_all_eval(
        model=model, dataset=hc_dataset, subset_size=cfg.subset_size, device=device
    )
    hc_results = {
        "mrr": hc_mrr,
        "top_k_mrr": hc_top_k_mrr,
        "ranks": hc_ranks_dict,
        "loss": hc_loss,
        "num_hc_presets": num_hc_presets,
    }

    log.info(f"Computing evaluation metrics on {num_hc_presets} random presets (one-vs-all)...")
    rnd_sub_mrr, rnd_sub_top_k_mrr, _, rnd_sub_loss = one_vs_all_eval(
        model=model, dataset=rnd_dataset, subset_size=num_hc_presets, device=device
    )
    rnd_sub_results = {
        "mrr": rnd_sub_mrr,
        "top_k_mrr": rnd_sub_top_k_mrr,
        "loss": rnd_sub_loss,
    }

    log.info(f"Computing evaluation metrics on {len(rnd_dataset)} random presets (non-overlapping)...")
    rnd_mrr, rnd_top_k_mrr, rnd_ranks_dict, rnd_loss = non_overlapping_eval(
        model=model, dataset=rnd_dataset, num_ranks=256, device=device
    )
    rnd_results = {
        "mrr": rnd_mrr,
        "top_k_mrr": rnd_top_k_mrr,
        "ranks": rnd_ranks_dict,
        "loss": rnd_loss,
        "num_rnd_presets": len(rnd_dataset),
    }

    results = {
        "hc": hc_results,
        "rnd": rnd_results,
        "rnd_sub": rnd_sub_results,
    }

    ##### Logging results
    object_dict = {
        "cfg": cfg,
        "model": m_preset,
        "dataset_cfg": hc_dataset.configs_dict,
        "results": results,
    }

    if cfg.get("wandb"):
        log.info("Logging hyperparameters...")
        eval_logger(object_dict=object_dict, run=run)
        wandb.finish()  # required for hydra multirun

    with open(Path(HydraConfig.get().runtime.output_dir) / "results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    evaluate()  # pylint: disable=no-value-for-parameter
