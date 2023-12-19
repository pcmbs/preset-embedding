"""
Module implementing embedding layers for the preset encoder.
"""
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

        self.cat_params_raw_values = [
            torch.tensor(cat_values, dtype=torch.float32)
            for cat_values, _ in preset_helper.grouped_used_cat_params.items()
        ]
        self.cat_params_idx = [
            torch.tensor(indices, dtype=torch.long)
            for _, indices in preset_helper.grouped_used_cat_params.items()
        ]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_params_raw_values[0].device != device:
            self.cat_params_raw_values = [param.to(device) for param in self.cat_params_raw_values]
            self.cat_params_idx = [param.to(device) for param in self.cat_params_idx]

        for cat_values, indices in zip(self.cat_params_raw_values, self.cat_params_idx):
            x[..., indices] = cat_values[x[..., indices].to(torch.long)]
        return x
