# pylint: disable=E1102
"""
Module implementing embedding layers for the preset encoder.
"""
from typing import Tuple
import torch
from torch import nn

from utils.synth import PresetHelper


class FeatureNormalizer(nn.Module):
    """Normalize all parameter values in the range [-1,1]."""

    def __init__(self, in_features: int) -> None:
        """
        Normalize all parameter values in the range [-1,1]
        assuming all parameter values are in the range [0,1].
        """
        super().__init__()
        self.in_features = in_features
        self.register_buffer("scaler", torch.empty(1), persistent=False)
        self.register_buffer("center", torch.empty(1), persistent=False)

    @property
    def out_length(self) -> int:
        return self.in_features

    @property
    def embedding_dim(self) -> int:
        return 1

    def init_weights(self) -> None:
        nn.init.constant_(self.scaler, 2.0)
        nn.init.constant_(self.center, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assuming all parameter values in the range [0,1]
        x = x * self.scaler - self.center
        return x


class RawParameters(nn.Module):
    """Use raw parameter values in the range [0,1] given the category indices of categorical synth parameters."""

    def __init__(self, preset_helper: PresetHelper) -> None:
        """
        Use raw parameter values in the range [0,1] given the category indices of categorical synth parameters.

        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """

        super().__init__()
        self._out_length = preset_helper.num_used_parameters
        self._grouped_used_parameters = preset_helper.grouped_used_parameters
        self.cat_parameters = self._group_cat_parameters_per_values()

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return 1

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_parameters[0][0].device != device:
            self.cat_parameters = [(val.to(device), idx.to(device)) for (val, idx) in self.cat_parameters]

        for cat_values, indices in self.cat_parameters:
            x[..., indices] = cat_values[x[..., indices].to(torch.long)]
        return x

    def _group_cat_parameters_per_values(self):
        cat_parameters_val_dict = {}
        for (cat_values, _), indices in self._grouped_used_parameters["discrete"]["cat"].items():
            # create copy of the indices list to not modify the original ones
            indices = indices.copy()
            if cat_values in cat_parameters_val_dict:
                cat_parameters_val_dict[cat_values] += indices
            else:
                cat_parameters_val_dict[cat_values] = indices

        return [
            (torch.tensor(cat_values, dtype=torch.float32), torch.tensor(indices, dtype=torch.long))
            for cat_values, indices in cat_parameters_val_dict.items()
        ]


class OneHotEncoding(nn.Module):
    """One-hot encoding of categorical parameters."""

    def __init__(self, preset_helper: PresetHelper) -> None:
        """
        One-hot encoding of categorical parameters.

        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """
        super().__init__()

        self._grouped_used_parameters = preset_helper.grouped_used_parameters

        # used numerical and binary parameters indices
        self.noncat_idx = torch.tensor(preset_helper.used_noncat_parameters_idx, dtype=torch.long)
        self.num_noncat = len(self.noncat_idx)

        # used categorical parameters
        self.cat_idx = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        self.cat_offsets, self.total_num_cat = self._compute_cat_infos(preset_helper)

        self._out_length = self.total_num_cat + self.num_noncat

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return 1

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_offsets.device != device:
            self.cat_offsets = self.cat_offsets.to(device)

        oh_enc = torch.zeros(x.shape[0], self.out_length, device=device)

        # Assign noncat parameters
        oh_enc[:, : self.num_noncat] = x[:, self.noncat_idx]

        # Calculate and assign ones for categorical
        ones_idx = x[:, self.cat_idx].to(dtype=torch.long) + self.cat_offsets + self.num_noncat
        oh_enc.scatter_(1, ones_idx, 1)
        # equivalent to oh_enc[torch.arange(x.shape[0]).unsqueeze(1), ones_idx] = 1

        return oh_enc

    def _compute_cat_infos(self, preset_helper: PresetHelper) -> Tuple[torch.Tensor, int]:
        """
        Compute the offsets for each categorical parameter and the total number of categories
        (i.e., sum over all categorical parameters' cardinality).

        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.

        Returns
        - `cat_offsets` (torch.Tensor): the offsets for each categorical parameter as a list cat_offsets[cat_param_idx] = offset.
        - `total_num_cat` (int):  total number of categories.
        """
        cat_offsets = []
        offset = 0
        for (cat_values, _), indices in preset_helper.grouped_used_parameters["discrete"]["cat"].items():
            for _ in indices:
                cat_offsets.append(offset)
                offset += len(cat_values)
        total_num_cat = offset
        cat_offsets = torch.tensor(cat_offsets, dtype=torch.long)
        return cat_offsets, total_num_cat


if __name__ == "__main__":
    import os
    from pathlib import Path
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader

    from data.datasets import SynthDatasetPkl

    SYNTH = "tal"
    BATCH_SIZE = 512

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATASET_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"

    if SYNTH == "tal":
        DATASET_PATH = DATASET_FOLDER / "talnm_mn04_size=65536_seed=45858_dev_val_v1"
        PARAMETERS_TO_EXCLUDE_STR = (
            "master_volume",
            "voices",
            "lfo_1_sync",
            "lfo_1_keytrigger",
            "lfo_2_sync",
            "lfo_2_keytrigger",
            "envelope*",
            "portamento*",
            "pitchwheel*",
            "delay*",
        )

        p_helper = PresetHelper("talnm", PARAMETERS_TO_EXCLUDE_STR)

    if SYNTH == "diva":
        PARAMETERS_TO_EXCLUDE_STR = (
            "main:output",
            "vcc:*",
            "opt:*",
            "scope1:*",
            "clk:*",
            "arp:*",
            "plate1:*",
            "delay1:*",
            "chrs2:*",
            "phase2:*",
            "rtary2:*",
            "*keyfollow",
            "*velocity",
            "env1:model",
            "env2:model",
            "*trigger",
            "*release_on",
            "env1:quantise",
            "env2:quantise",
            "env1:curve",
            "env2:curve",
            "lfo1:sync",
            "lfo2:sync",
            "lfo1:restart",
            "lfo2:restart",
            "mod:rectifysource",
            "mod:invertsource",
            "mod:addsource*",
            "*revision",
            "vca:pan",
            "vca:volume",
            "vca:vca",
            "vca:panmodulation",
            "vca:panmoddepth",
            "vca:mode",
            "vca:offset",
        )
        p_helper = PresetHelper("diva", PARAMETERS_TO_EXCLUDE_STR)

    dataset = SynthDatasetPkl(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    oh = OneHotEncoding(p_helper)
    oh.to(DEVICE)

    for params, _ in loader:
        oh(params.to(DEVICE))
        break
    print("")
