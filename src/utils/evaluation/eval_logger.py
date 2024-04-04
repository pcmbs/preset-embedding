"""
Module implementing a function to log hyperparameters.

Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/logging_utils.py 
"""

import logging
import os
from typing import Any, Dict
from omegaconf import OmegaConf
import wandb

log = logging.getLogger(__name__)


def eval_logger(object_dict: Dict[str, Any], run: wandb.sdk.wandb_run.Run) -> None:
    """Log infos & hps related to the evaluation.

    Args
    - `object_dict`: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: nn.Module for the preset encoder model.
        - `"dataset_cfg"`: A DictConfig object containing the dataset config.
        - `"results"`: A dictionary containing the evaluation results.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    dataset_cfg = object_dict["dataset_cfg"]
    hc_results = object_dict["results"]["hc"]
    rnd_results = object_dict["results"]["rnd"]
    rnd_sub_results = object_dict["results"]["rnd_sub"]

    # General hyperparameters
    hparams["seed"] = cfg.get("seed")

    # Data related hyperparameters
    hparams["data/synth"] = cfg["synth"]["name"]
    hparams["data/num_hc_presets"] = hc_results["num_hc_presets"]
    hparams["data/num_rnd_presets"] = rnd_results["num_rnd_presets"]
    hparams["data/excluded_params"] = dataset_cfg["params_to_exclude"]
    hparams["data/num_used_params"] = dataset_cfg["num_used_params"]
    hparams["data/render_duration_in_sec"] = dataset_cfg["render_duration_in_sec"]
    hparams["data/midi_note"] = dataset_cfg["midi_note"]
    hparams["data/midi_velocity"] = dataset_cfg["midi_velocity"]
    hparams["data/midi_duration_in_sec"] = dataset_cfg["midi_duration_in_sec"]
    hparams["data/audio_fe"] = dataset_cfg["audio_fe"]
    hparams["data/embedding_dim"] = dataset_cfg["num_outputs"]

    # Model hyperparameters
    hparams["model/type"] = cfg["model"]["type"]
    hparams["model/num_parameters"] = object_dict["model"].num_parameters
    hparams["model/ckpt_name"] = cfg["model"]["ckpt_name"]
    hparams["num_train_epochs"] = cfg["model"]["num_train_epochs"]

    for k, v in cfg["model"]["cfg"].items():
        if k == "_target_":
            hparams["model/name"] = v.split(".")[-1]
        elif k in ["embedding_kwargs", "block_kwargs"]:
            for kk, vv in v.items():
                hparams[f"model/{k}/{kk}"] = vv
        else:
            hparams[f"model/{k}"] = v

    # Results
    metrics_dict = {}
    # Results from training
    metrics_dict["val"] = {
        "mrr": cfg["model"]["val_mrr"],
        "loss": cfg["model"]["val_loss"],
        "epoch": cfg["model"]["epoch"],
    }
    # Results on hand-crafted presets
    metrics_dict["hc"] = {"mrr": hc_results["mrr"], "loss": hc_results["loss"]}
    for k, mrr in enumerate(hc_results["top_k_mrr"]):
        metrics_dict["hc"][f"top_{k+1}_mrr"] = mrr
    # Results on random presets
    metrics_dict["rnd"] = {"mrr": rnd_results["mrr"], "loss": rnd_results["loss"]}
    for k, mrr in enumerate(rnd_results["top_k_mrr"]):
        metrics_dict["rnd"][f"top_{k+1}_mrr"] = mrr
    # Results on random subsets of presets
    metrics_dict["rnd_sub"] = {"mrr": rnd_sub_results["mrr"], "loss": rnd_sub_results["loss"]}
    for k, mrr in enumerate(rnd_sub_results["top_k_mrr"]):
        metrics_dict["rnd_sub"][f"top_{k+1}_mrr"] = mrr

    raw_table = []
    for k, v in metrics_dict.items():
        for kk, vv in v.items():
            if kk in ["loss", "top_1_mrr", "top_5_mrr"]:
                raw_table.append((k, kk, vv))
    columns = ["dataset", "metric", "value"]

    # log metrics
    run.log({"metrics": metrics_dict})  # log instead of summary to be able to create bar charts in wandb UI
    run.summary["grouped_metrics"] = wandb.Table(data=raw_table, columns=columns)
    wandb.config.update(hparams)

    # hydra config is saved under <project_name>/Runs/<run_id>/Files/.hydra
    wandb.save(
        glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
        base_path=cfg["paths"].get("output_dir"),
    )
