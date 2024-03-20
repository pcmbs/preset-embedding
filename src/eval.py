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

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from torch import nn
from omegaconf import DictConfig
import torch

from data.datasets import SynthDatasetPkl
from models.lit_module import PresetEmbeddingLitModule
from utils.evaluation import hc_eval_mrr, compute_l1, rnd_presets_eval
from utils.logging import RankedLogger
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="eval.yaml")
def evaluate(cfg: DictConfig) -> None:
    """
    MRR evaluation for the model.

    Args
    - `cfg`: A DictConfig configuration composed by Hydra.

    Return

    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating hand-crafted presets Dataset: {cfg.synth.dataset_path}")
    presets_dataset = SynthDatasetPkl(cfg.synth.dataset_path)
    log.info(f"Instantiating random presets Dataset: {cfg.synth.rnd_dataset_path}")
    rnd_presets_dataset = SynthDatasetPkl(cfg.synth.rnd_dataset_path)
    assert (
        presets_dataset.configs_dict["params_to_exclude"]
        == rnd_presets_dataset.configs_dict["params_to_exclude"]
    )

    log.info(f"Instantiating Preset Helper for synth {presets_dataset.synth_name} and excluded params:")
    log.info(f"{presets_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=presets_dataset.synth_name,
        parameters_to_exclude=presets_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.model.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.model.cfg, out_features=presets_dataset.embedding_dim, preset_helper=preset_helper
    )

    log.info(f"Loading checkpoint from {cfg.ckpt_path}...")
    model = PresetEmbeddingLitModule.load_from_checkpoint(cfg.ckpt_path, preset_encoder=m_preset)
    model.freeze()

    log.info("Computing evaluation metrics on the hand-crafted presets dataset...")
    mrr_score, top_k_mrr, ranks_dict = hc_eval_mrr(model=model, dataset=presets_dataset)
    l1_loss = compute_l1(model=model, dataset=presets_dataset)

    results = {
        "mrr": mrr_score,
        "top_k_mrr": top_k_mrr,
        "ranks": ranks_dict,
        "l1": l1_loss,
        "synth": presets_dataset.synth_name,
        "dataset_name": cfg.synth.name,
        "num_hc_presets": len(presets_dataset),
        "model_name": cfg.model.name,
    }

    log.info("Computing evaluation metrics on the random presets dataset...")
    _, _, _, _ = rnd_presets_eval(model=model, dataset=rnd_presets_dataset, num_ranks=256)

    with open(Path(HydraConfig.get().runtime.output_dir) / "mrr_results.pkl", "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    # import sys
    # args = ["src/eval.py", "model=mlp_relu_oh_talnm", "task_name=debug"]

    # sys.argv = args

    # gettrace = getattr(sys, "gettrace", None)
    # if gettrace():
    #     sys.argv = args

    evaluate()  # pylint: disable=no-value-for-parameter
