import os
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from utils.synth import PresetHelper, PresetRenderer

load_dotenv()  # take environment variables from .env

TALNM_PATH = os.environ["TALNM_PATH"]
DIVA_PATH = os.environ["DIVA_PATH"]
DEXED_PATH = os.environ["DEXED_PATH"]


class SynthDataset(Dataset):
    """
    Map-style dataset for generating random presets for a given syntheiszer using DawDreamer for rendering.
    """

    MAX_SEED_VALUE = 2**64 - 1  # 18_446_744_073_709_551_615
    OFFSET_COEFFICIENT = 100_000_000_000

    def __init__(
        self,
        preset_helper: PresetHelper,
        dataset_size: int = 100_000,
        seed_offset: int = 0,
        sample_rate: int = 44_100,
        render_duration_in_sec: float = 5.0,
        rms_range: Tuple[float, float] = (0.01, 1.0),
        midi_note: Optional[int] = 60,
        midi_velocity: Optional[int] = 100,
        midi_duration_in_sec: Optional[float] = 2,
        path_to_plugin: Optional[str | Path] = None,
    ):
        """
        Args
        - `preset_helper` (PresetHelper): Instance of the PresetHelper class, containing information about
        the parameters of a given synthesizer. This is used to generate random presets.
        - `dataset_size` (int): Total number of data samples that will be generated in an epoch.
        Can be set to a really big number for iterative training. In that case, the DataLoader's shuffle
        parameter must be set to False in order to use a SequentialSampler instead of a RandomSampler
        which could cause OOM errors due to the torch.randperm(dataset_size) call, since e.g., a dataset
        size of 10M will require 80MB of RAM, while a dataset size of 100M will require 800MB of RAM.
        Note that setting dataset_size!=100B will make the retrieved index to always be -1 since getting the
        index of the last sampled data is not useful in epoch-based training based on a RandomSampler
        (default: 100_000)
        - `seed_offset` (int): Positive offset (multiplied by 10B) to be added to the RNG seed which
        corresponds to the first data sample's index, since the global training step (+ this offset)
        is used to set the RNG seed for each data point. This allows to generate 184_467_440 unique
        sequences of 100B presets each. (default: 0)
        - `sample_rate` (int): Sample rate of the audio to generate. (default: 44_100)
        - `render_duration_in_sec` (float): Rendering duration in seconds. (default: 4.0)
        - `midi_note` (Optional[int]): Midi note use to generate all data sample. (default: 60)
        - `midi_velocity` (Optional[int]): MIDI note velocity to generate all data sample. (default: 110)
        - `midi_duration_in_sec` (Optional[float]): MIDI note duration in seconds to generate all data sample.
        (default: 2.0)
        - `rms_range` (Tuple[float, float]): acceptable audio RMS range. If a generated audio is out of this
        range, a new preset will be generated. (default: (0.01, 1.0))
        - `path_to_plugin` (Optional[str]): Path to the plugin. If None (default), it will look for it in
        <project-folder>/data based on preset_helper.synth_name.

        """
        self.preset_helper = preset_helper

        if not isinstance(dataset_size, int) and dataset_size <= 0:
            raise ValueError(f"dataset_size must be a positive integer, but got dataset_size={dataset_size}")
        if not isinstance(seed_offset, int) and seed_offset < 0:
            raise ValueError(f"seed_offset must be a positive integer, but got seed_offset={seed_offset}")

        if seed_offset * self.OFFSET_COEFFICIENT + dataset_size > self.MAX_SEED_VALUE:
            raise ValueError(
                f"start_seed*100B + dataset_size must be <= {self.MAX_SEED_VALUE}, but got start_seed={seed_offset} and dataset_size={dataset_size}"
            )

        self.dataset_size = dataset_size
        self.seed_offset = int(seed_offset * self.OFFSET_COEFFICIENT)

        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.midi_duration_in_sec = midi_duration_in_sec
        self.render_duration_in_sec = render_duration_in_sec

        self.rms_range = rms_range

        if path_to_plugin is None:
            if preset_helper.synth_name == "talnm":
                path_to_plugin = TALNM_PATH
            elif preset_helper.synth_name == "dexed":
                path_to_plugin = DEXED_PATH
            elif preset_helper.synth_name == "diva":
                path_to_plugin = DIVA_PATH
            else:
                raise NotImplementedError()

        path_to_plugin = str(path_to_plugin)

        self.renderer = PresetRenderer(
            synth_path=path_to_plugin,
            sample_rate=sample_rate,
            render_duration_in_sec=render_duration_in_sec,
            convert_to_mono=True,
            normalize_audio=False,
            fadeout_in_sec=0.1,
        )

        # set not used parameters to default values (for safety)
        self.renderer.set_parameters(
            self.preset_helper.excl_parameters_idx, self.preset_helper.excl_parameters_val
        )

    @property
    def synth_name(self):
        """Return the name of the synthesizer."""
        return self.preset_helper.synth_name

    @property
    def rnd_sampling_info(self):
        """Return the description of the random sampling parameters."""
        return self.preset_helper.grouped_used_parameters

    @property
    def num_used_parameters(self):
        """Return the number of used synthesizer parameters."""
        return self.preset_helper.num_used_parameters

    @property
    def used_parameters_description(self) -> List[Tuple[int, str]]:
        """Return the description of the used synthesizer parameters as a
        list of tuple (<idx>, <synth-param-idx>, <synth-param-name>, <synth-param-type>)."""
        return self.preset_helper.used_parameters_description

    @property
    def sample_rate(self):
        """Return the sample rate of the audio to generate."""
        return self.renderer.sample_rate

    @property
    def index_range(self) -> Tuple[int, int]:
        """Return the range of seeds that will be used to generate data."""
        return (self.seed_offset, self.seed_offset + self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        rng_cpu = torch.Generator(device="cpu")
        rng_cpu.manual_seed(self.seed_offset + idx)

        rms_out = self.rms_range[0] - 1.0  # get in the loop

        # generate random preset until the audio rms value is in an acceptable range
        while not self.rms_range[0] < rms_out < self.rms_range[1]:
            # generate random synth parameter values
            synth_parameters, cat_parameters_int, cat_parameters_idx = self._sample_parameter_values(
                self.rnd_sampling_info, rng_cpu
            )
            # set synth parameters
            self.renderer.set_parameters(self.preset_helper.used_parameters_idx, synth_parameters)
            # set midi parameters
            self.renderer.set_midi_parameters(self.midi_note, self.midi_velocity, self.midi_duration_in_sec)
            # render audio
            audio_out = torch.from_numpy(self.renderer.render_note())
            # check rms
            rms_out = torch.sqrt(torch.mean(torch.square(audio_out))).item()

        # index only required to resume training for pseudo-infinite dataset
        idx = idx if self.dataset_size == 100_000_000_000 else -1

        # replace raw categorical parameter values by the category index
        synth_parameters[cat_parameters_idx] = torch.tensor(cat_parameters_int, dtype=torch.float32)

        return synth_parameters, audio_out, idx

    def _sample_parameter_values(self, rnd_sampling_info: dict, rng: torch.Generator):
        rnd_parameters = torch.empty(self.num_used_parameters, dtype=torch.float32)
        cat_parameters_int = []
        cat_parameters_idx = []

        for interval, indices in rnd_sampling_info["continuous"].items():
            rnd_parameters[indices] = torch.empty(len(indices)).uniform_(*interval, generator=rng)

        for param_type, sampling_dict in rnd_sampling_info["discrete"].items():
            for (values, weights), indices in sampling_dict.items():
                sampled_idx = torch.multinomial(
                    torch.tensor(weights), len(indices), replacement=True, generator=rng
                )
                rnd_parameters[indices] = torch.tensor(values, dtype=torch.float32)[sampled_idx]
                if param_type == "cat":
                    cat_parameters_int += sampled_idx.tolist()
                    cat_parameters_idx += indices

        return rnd_parameters, cat_parameters_int, cat_parameters_idx

    # def _sample_parameter_values(self, rnd_sampling_info: dict, rng: torch.Generator):
    #     rnd_parameters = torch.empty(self.num_used_parameters, dtype=torch.float32)
    #     cat_parameters_int = []
    #     cat_parameters_idx = []

    #     for interval, indices in rnd_sampling_info["num"].items():
    #         rnd_parameters[indices] = torch.empty(len(indices)).uniform_(*interval, generator=rng)

    #     for (cat_values, cat_weights), indices in rnd_sampling_info["cat"].items():
    #         sampled_cat_idx = torch.multinomial(
    #             torch.tensor(cat_weights), len(indices), replacement=True, generator=rng
    #         )
    #         # sampled_cat_idx = torch.randint(0, len(cat_values), (len(indices),), generator=rng)
    #         rnd_parameters[indices] = torch.tensor(cat_values, dtype=torch.float32)[sampled_cat_idx]
    #         cat_parameters_int += sampled_cat_idx.tolist()
    #         cat_parameters_idx += indices

    #     rnd_parameters[rnd_sampling_info["bin"]] = torch.bernoulli(
    #         torch.full((len(rnd_sampling_info["bin"]),), 0.5), generator=rng
    #     )

    #     return rnd_parameters, cat_parameters_int, cat_parameters_idx


if __name__ == "__main__":
    print("")
