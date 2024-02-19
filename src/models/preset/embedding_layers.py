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
    def embedding_dim(self) -> int:
        return self.in_features

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
        self._embedding_dim = preset_helper.num_used_params

        self.cat_params = [
            (torch.tensor(cat_values, dtype=torch.float32), torch.tensor(indices, dtype=torch.long))
            for cat_values, indices in preset_helper.cat_params_val_dict.items()
        ]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_params[0][0].device != device:
            self.cat_params = [(val.to(device), idx.to(device)) for (val, idx) in self.cat_params]

        for cat_values, indices in self.cat_params:
            x[..., indices] = cat_values[x[..., indices].to(torch.long)]
        return x


class OneHotEncoding(nn.Module):
    """One-hot encoding of categorical parameters."""

    def __init__(self, preset_helper: PresetHelper) -> None:
        """
        One-hot encoding of categorical parameters.

        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """
        super().__init__()

        # used numerical and binary parameters indices
        self.non_cat_params = preset_helper.used_num_params_idx + preset_helper.used_bin_params_idx
        self.num_non_cat_params = len(self.non_cat_params)

        self.cat_params = [
            (cat_card, torch.tensor(indices, dtype=torch.long))
            for cat_card, indices in preset_helper.cat_params_card_dict.items()
        ]

        self._embedding_dim = int(
            sum(card * len(indices) for card, indices in self.cat_params) + self.num_non_cat_params
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_params[0][1].device != device:
            self.cat_params = [(card, idx.to(device)) for (card, idx) in self.cat_params]

        emb = torch.empty(x.shape[0], self.embedding_dim, device=device)
        emb[..., : self.num_non_cat_params] = x[..., self.non_cat_params]

        # Calculate and assign categorical embeddings
        emb_index = self.num_non_cat_params
        for card, idx in self.cat_params:
            cat_emb = F.one_hot(x[..., idx].to(torch.long), num_classes=card).view(x.shape[0], -1)
            emb[..., emb_index : emb_index + cat_emb.shape[1]] = cat_emb
            emb_index += cat_emb.shape[1]

        return emb


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data.datasets import SynthDataset

    NUM_SAMPLES = 10
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

    p_helper = PresetHelper("tal_noisemaker", PARAMETERS_TO_EXCLUDE_STR)
    dataset = SynthDataset(p_helper, NUM_SAMPLES, seed_offset=5423)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
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
