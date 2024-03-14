# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
# pylint: disable=E1120:no-value-for-parameter
"""
Evaluation script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
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
from utils.evaluation import eval_mrr
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
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating evaluation Dataset: {cfg.dataset.path}")
    eval_dataset = SynthDatasetPkl(cfg.dataset.path)

    log.info(f"Instantiating Preset Helper for synth {eval_dataset.synth_name} and excluded params:")
    log.info(f"{eval_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=eval_dataset.synth_name,
        parameters_to_exclude=eval_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.model.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.model.cfg, out_features=eval_dataset.embedding_dim, preset_helper=preset_helper
    )

    log.info(f"Loading checkpoint from {cfg.ckpt_path}...")
    model = PresetEmbeddingLitModule.load_from_checkpoint(cfg.ckpt_path, preset_encoder=m_preset)
    model.freeze()

    log.info("Running MRR evaluation...")
    mrr_score, ranks_dict = eval_mrr(model=model, dataset=eval_dataset)

    # TODO: add L1 distance

    results = {
        "mrr": mrr_score,
        "ranks": ranks_dict,
        "synth": eval_dataset.synth_name,
        "dataset_name": cfg.dataset.path,
        "num_presets": len(eval_dataset),
        "model_name": cfg.model.name,
        "ckpt_name": cfg.model.ckpt_name,
    }

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
