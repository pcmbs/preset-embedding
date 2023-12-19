# pylint: disable=W1203
"""
Module to be run from the command line used to render and export presets
(and optionaly the associated synthesizer parameters) from a given synthesizer and settings. 

See `export_dataset_wav_cfg.yaml` for more details.

The results are exported to
`<project-root>/<export-relative-path>/<synth>_<audio_fe>_<dataset-size>_<seed-offset>_wav_<tag>`
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from scipy.io import wavfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import SynthDataset
from utils.synth import PresetHelper

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/export", config_name="dataset_wav")
def export_dataset_wav(cfg: DictConfig) -> None:
    configs_dict = {
        "synth": cfg.synth,
        "params_to_exclude": cfg.parameters_to_exclude_str,
        "num_samples": cfg.num_samples,
        "seed_offset": cfg.seed_offset,
        "sample_rate": cfg.sample_rate,
        "render_duration_in_sec": cfg.render_duration_in_sec,
        "midi_note": cfg.midi_note,
        "midi_velocity": cfg.midi_velocity,
        "midi_duration_in_sec": cfg.midi_duration_in_sec,
    }

    log.info("Configs")
    for k, v in configs_dict.items():
        log.info(f"{k}: {v}")

    with open(Path.cwd() / "_configs.pkl", "wb") as f:
        torch.save(configs_dict, f)

    p_helper = PresetHelper(synth_name=cfg.synth, params_to_exclude_str=cfg.parameters_to_exclude_str)

    dataset = SynthDataset(
        preset_helper=p_helper,
        dataset_size=cfg.num_samples,
        seed_offset=cfg.seed_offset,
        sample_rate=cfg.sample_rate,
        render_duration_in_sec=cfg.render_duration_in_sec,
        midi_note=cfg.midi_note,
        midi_velocity=cfg.midi_velocity,
        midi_duration_in_sec=cfg.midi_duration_in_sec,
    )

    log.info("")
    log.info("Synth parameters description")
    for param in dataset.used_params_description:
        log.info(param)

    with open(Path.cwd() / "_synth_parameters_description.pkl", "wb") as f:
        torch.save(dataset.used_params_description, f)

    dataloader = DataLoader(
        dataset,
        batch_size=1 if cfg.num_samples < 128 else 32,
        num_workers=0 if cfg.num_samples < 128 else 8,
    )

    pbar = tqdm(
        dataloader,
        total=cfg.num_samples if cfg.num_samples < 128 else cfg.num_samples // dataloader.batch_size,
        dynamic_ncols=True,
    )

    for i, (params_batch, audio_batch, _) in enumerate(pbar):
        if i == cfg.num_samples:
            break

        for j, (params, audio) in enumerate(zip(params_batch, audio_batch)):
            wavfile.write(Path.cwd() / f"{(i+1)*j}.wav", cfg.sample_rate, audio.numpy().T)

            if cfg.export_synth_params:
                with open(Path.cwd() / f"{(i+1)*j}.pkl", "wb") as f:
                    torch.save(params, f)


if __name__ == "__main__":
    export_dataset_wav()  # pylint: disable=E1120:no-value-for-parameter̦
