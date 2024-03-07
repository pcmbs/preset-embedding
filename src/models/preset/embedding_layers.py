# pylint: disable=E1102
"""
Module implementing embedding layers for the preset encoder.
"""
import torch
from torch import nn
import torch.nn.functional as F

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
    def out_lenght(self) -> int:
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
        self.noncat_parameters = preset_helper.used_noncat_parameters_idx
        self.num_noncat_parameters = len(self.noncat_parameters)

        self.cat_parameters = self._group_cat_parameters_per_cardinality()

        self._out_length = int(
            sum(card * len(indices) for card, indices in self.cat_parameters) + self.num_noncat_parameters
        )

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
        if self.cat_parameters[0][1].device != device:
            self.cat_parameters = [(card, idx.to(device)) for (card, idx) in self.cat_parameters]

        emb = torch.empty(x.shape[0], self.out_length, device=device)
        emb[..., : self.num_noncat_parameters] = x[..., self.noncat_parameters]

        # Calculate and assign categorical embeddings
        emb_index = self.num_noncat_parameters
        for card, idx in self.cat_parameters:
            cat_emb = F.one_hot(x[..., idx].to(torch.long), num_classes=card).view(x.shape[0], -1)
            emb[..., emb_index : emb_index + cat_emb.shape[1]] = cat_emb
            emb_index += cat_emb.shape[1]

        return emb

    def _group_cat_parameters_per_cardinality(self):
        cat_parameters_card_dict = {}
        for (cat_values, _), indices in self._grouped_used_parameters["discrete"]["cat"].items():
            # create copy of the indices list to not modify the original ones
            indices = indices.copy()
            num_values = len(cat_values)
            if num_values in cat_parameters_card_dict:
                cat_parameters_card_dict[num_values] += indices
            else:
                cat_parameters_card_dict[num_values] = indices

        return [
            (cat_card, torch.tensor(indices, dtype=torch.long))
            for cat_card, indices in cat_parameters_card_dict.items()
        ]
        # return deepcopy(cat_params_card_dict)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data.datasets import SynthDataset

    SYNTH = "diva"
    NUM_SAMPLES = 1

    if SYNTH == "tal":
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

    dataset = SynthDataset(p_helper, NUM_SAMPLES, seed_offset=5423)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    raw = RawParameters(p_helper)
    one_hot_enc = OneHotEncoding(p_helper)

    for sample in loader:
        # raw_emb = raw(sample[0])
        # print(raw_emb)
        # print(raw_emb.shape)
        one_hot_emb = one_hot_enc(sample[0])
        print(one_hot_emb)
        print(one_hot_emb.shape)

    print("")
