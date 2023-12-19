# pylint: disable=W1203
"""
Module to be run from the command line used to generate audio embeddings
from a given audio model together with the corresponding synth parameters. 

See `export_dataset_pkl_cfg.yaml` for more details.

The results are exported to
`<project-root>/<export-relative-path>/<synth>_<audio_fe>_<dataset-size>_<seed-offset>_pkl_<tag>`
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import wavfile

from data.datasets import SynthDataset
from models import audio as audio_models
from utils.synth import PresetHelper

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../../configs/export", config_name="dataset_pkl")
def export_dataset_pkl(cfg: DictConfig) -> None:
    # instantiate audio model here to store attributes in configs_dict
    audio_fe = getattr(audio_models, cfg.audio_fe)()

    if cfg.export_audio:
        audio_path = Path.cwd() / "audio_first_batch"
        audio_path.mkdir(parents=True, exist_ok=True)

    configs_dict = {
        "synth": cfg.synth,
        "params_to_exclude": cfg.parameters_to_exclude_str,
        "dataset_size": cfg.dataset_size,
        "seed_offset": cfg.seed_offset,
        "render_duration_in_sec": cfg.render_duration_in_sec,
        "midi_note": cfg.midi_note,
        "midi_velocity": cfg.midi_velocity,
        "midi_duration_in_sec": cfg.midi_duration_in_sec,
        "audio_fe": cfg.audio_fe,
        "sample_rate": audio_fe.sample_rate,
        "num_outputs": audio_fe.out_features,
    }

    log.info("Configs")
    for k, v in configs_dict.items():
        log.info(f"{k}: {v}")

    with open(Path.cwd() / "configs.pkl", "wb") as f:
        torch.save(configs_dict, f)

    p_helper = PresetHelper(synth_name=cfg.synth, params_to_exclude_str=cfg.parameters_to_exclude_str)

    dataset = SynthDataset(
        preset_helper=p_helper,
        dataset_size=cfg.dataset_size,
        seed_offset=cfg.seed_offset,
        sample_rate=audio_fe.sample_rate,
        render_duration_in_sec=cfg.render_duration_in_sec,
        midi_note=cfg.midi_note,
        midi_velocity=cfg.midi_velocity,
        midi_duration_in_sec=cfg.midi_duration_in_sec,
    )

    log.info("")
    log.info("Synth parameters description")
    for param in dataset.used_params_description:
        log.info(param)

    with open(Path.cwd() / "synth_parameters_description.pkl", "wb") as f:
        torch.save(dataset.used_params_description, f)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    pbar = tqdm(dataloader, total=int(cfg.dataset_size / cfg.batch_size), dynamic_ncols=True)

    audio_emb_to_pickle = torch.empty((cfg.dataset_size, audio_fe.out_features), device="cpu")
    synth_params_to_pickle = torch.empty((cfg.dataset_size, dataset.num_used_parameters), device="cpu")

    audio_fe.to(DEVICE)
    audio_fe.eval()

    for i, (params, audio, _) in enumerate(pbar):
        audio = audio.to(DEVICE)
        with torch.no_grad():
            audio_emb = audio_fe(audio)

        audio_emb_to_pickle[i * cfg.batch_size : (i + 1) * cfg.batch_size] = audio_emb.cpu()
        synth_params_to_pickle[i * cfg.batch_size : (i + 1) * cfg.batch_size] = params.cpu()

        if cfg.export_audio:
            for i, sample in enumerate(audio):
                sample = sample.cpu().numpy()
                wavfile.write(audio_path / f"{i}.wav", audio_fe.sample_rate, sample.T)

    with open(Path.cwd() / "audio_embeddings.pkl", "wb") as f:
        torch.save(audio_emb_to_pickle, f)

    with open(Path.cwd() / "synth_params.pkl", "wb") as f:
        torch.save(synth_params_to_pickle, f)


if __name__ == "__main__":
    export_dataset_pkl()  # pylint: disable=E1120:no-value-for-parameter̦
