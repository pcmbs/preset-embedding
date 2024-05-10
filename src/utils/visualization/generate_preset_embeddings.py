# pylint: disable=E1120:no-value-for-parameter
import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.lit_module import PresetEmbeddingLitModule
from models.preset import model_zoo
from utils.synth import PresetHelper

load_dotenv()

# # should be one of ["dexed", "diva", "talnm"]
# synth = "dexed"
# # should be one of ["mlp_oh", "highway_oh", "highway_ft", "highway_ftgru", "tfm"]
# model = "mlp_oh"

# # dataset version and batch size to generate embeddings (can be left as such)
# dataset_version = 1
# batch_size = 512

# Required Paths
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_preset_embeddings(
    synth: str, model: str, batch_size: int = 512, dataset_version: int = 1
) -> None:
    # Load eval configs to retrieve model configs and ckpt name
    eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "eval" / "model" / f"{synth}_{model}.yaml")["model"]
    model_cfg = {k: v for k, v in eval_cfg["cfg"].items() if k != "_target_"}
    ckpt_name = eval_cfg["ckpt_name"]

    # path to the eval dataset
    data_dir = PROJECT_ROOT / "data" / "datasets" / "eval" / f"{synth}_mn04_eval_v{dataset_version}"

    with open(data_dir / "configs.pkl", "rb") as f:
        configs = torch.load(f)

    with open(data_dir / "synth_parameters.pkl", "rb") as f:
        synth_parameters = torch.load(f)

    preset_helper = PresetHelper(
        synth_name=synth,
        parameters_to_exclude=configs["params_to_exclude"],
    )

    m_preset = getattr(model_zoo, model)(
        **model_cfg, out_features=configs["num_outputs"], preset_helper=preset_helper
    )
    lit_model = PresetEmbeddingLitModule.load_from_checkpoint(
        str(PROJECT_ROOT / "checkpoints" / ckpt_name), preset_encoder=m_preset
    )
    lit_model.to(DEVICE)
    lit_model.freeze()

    preset_embeddings = torch.empty(len(synth_parameters), configs["num_outputs"])

    dataset = TensorDataset(synth_parameters)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch[0].to(DEVICE)
        with torch.no_grad():
            preset_embeddings[i * batch_size : (i + 1) * batch_size] = lit_model(batch)

    (data_dir / "preset_embeddings").mkdir(parents=True, exist_ok=True)
    with open(data_dir / "preset_embeddings" / f"{model}_embeddings.pkl", "wb") as f:
        torch.save(preset_embeddings, f)
