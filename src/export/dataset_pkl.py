# pylint: disable=W1203
"""
Module to be run from the command line used to generate audio embeddings
from a given audio model together with the corresponding synth parameters. 

See `export_dataset_pkl_cfg.yaml` for more details.

The results are exported to
`<project-root>/<export-relative-path>/<synth>_<audio_fe>_<dataset-size>_<seed-offset>_pkl_<tag>`

It is possible to interrupt the process at the end of the current iteration by pressing Ctrl+Z. 
Doing so will export a resume state file that will be used to resume the process. 

"""
import logging
import os
from pathlib import Path
import signal
import sys

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import wavfile

from data.datasets import SynthDataset
from models import audio as audio_models
from utils.data import SequentialSampler2
from utils.synth import PresetHelper

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESUME_STATE_FILE = "resume_state.pkl"

# flag to track if keyboard interrupt has been detected
# which is used to stop export at the end of the current batch.
INTERRUPTED = False


def batch_to_idx(batch_index: int, batch_size: int) -> int:
    """Convert a batch index to an iteration index."""
    return batch_index * batch_size


def sigstp_handler(signal, frame) -> None:
    """Handler for SIGSTP signal, interrupt the export process."""
    global INTERRUPTED  # pylint: disable=W0603
    print("Ctrl+Z detected, the export will be aborted at the end of the current iteration...")
    INTERRUPTED = True


# Set the SIGSTP signal handler
signal.signal(signal.SIGTSTP, sigstp_handler)


@hydra.main(version_base=None, config_path="../../configs/export", config_name="dataset_pkl")
def export_dataset_pkl(cfg: DictConfig) -> None:
    # instantiate audio model, preset helper and dataset
    audio_fe = getattr(audio_models, cfg.audio_fe)()
    audio_fe.to(DEVICE)
    audio_fe.eval()

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

    if (Path.cwd() / "resume_state.pkl").exists():
        with open(Path.cwd() / "resume_state.pkl", "rb") as f:
            saved_data = torch.load(f)
            start_index = saved_data["start_index"]
            audio_embeddings = saved_data["audio_embeddings"]
            synth_params = saved_data["synth_params"]

        log.info(
            f"resume_state.pkl found, resuming from batch {start_index}.\n"
            f"Number of remaining batches to be "
            f"generated: {(cfg.dataset_size - cfg.batch_size * start_index) // cfg.batch_size}",
        )

    else:
        start_index = 0

        if cfg.export_audio:
            audio_path = Path.cwd() / "audio"
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

        log.info("")
        log.info("Synth parameters description")
        for param in dataset.used_params_description:
            log.info(param)

        with open(Path.cwd() / "synth_parameters_description.pkl", "wb") as f:
            torch.save(dataset.used_params_description, f)

        audio_embeddings = torch.empty((cfg.dataset_size, audio_fe.out_features), device="cpu")
        synth_params = torch.empty((cfg.dataset_size, dataset.num_used_parameters), device="cpu")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        sampler=SequentialSampler2(dataset, start_idx=start_index * cfg.batch_size),
    )

    pbar = tqdm(
        dataloader,
        total=(cfg.dataset_size - cfg.batch_size * start_index) // cfg.batch_size,
        dynamic_ncols=True,
    )

    for i, (params, audio, _) in enumerate(pbar):
        i += start_index

        audio = audio.to(DEVICE)
        with torch.no_grad():
            audio_emb = audio_fe(audio)

        audio_embeddings[i * cfg.batch_size : (i + 1) * cfg.batch_size] = audio_emb.cpu()
        synth_params[i * cfg.batch_size : (i + 1) * cfg.batch_size] = params.cpu()

        if cfg.export_audio:
            for j, sample in enumerate(audio):
                sample = sample.cpu().numpy()
                wavfile.write(audio_path / f"{i+j}.wav", audio_fe.sample_rate, sample.T)

        if INTERRUPTED:
            current_index = i
            log.info(f"Finished generating batch {i}, interrupting generation...")
            break

    if INTERRUPTED:
        log.info("Saving resume_state.pkl file and exiting...")
        with open(Path.cwd() / "resume_state.pkl", "wb") as f:
            saved_data = {
                "start_index": current_index + 1,
                "audio_embeddings": audio_embeddings,
                "synth_params": synth_params,
            }
            torch.save(saved_data, f)
        sys.exit(0)

    log.info("All samples generated, saving audio embeddings and synth params tensors...")
    # Remove the interrupt file if export completes successfully
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)

    with open(Path.cwd() / "audio_embeddings.pkl", "wb") as f:
        torch.save(audio_embeddings, f)

    with open(Path.cwd() / "synth_params.pkl", "wb") as f:
        torch.save(synth_params, f)


if __name__ == "__main__":
    export_dataset_pkl()  # pylint: disable=E1120:no-value-for-parameter̦
