"""
Module implementing building blocks for the preset encoder.
"""

from typing import Optional
import torch
from torch import nn


class SelfNormalizingBlock(nn.Module):
    """Self-Normalizing MLP block as proposed in https://arxiv.org/pdf/1706.02515.pdf"""

    def __init__(
        self, in_features: int, out_features: int, hidden_features: Optional[int] = None, dropout_p=0.0
    ) -> None:
        """
        Self-Normalizing MLP block as proposed in https://arxiv.org/pdf/1706.02515.pdf.
        Original Pytorch implementation:
        https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features
        - `hidden_features` (int): number of hidden features. Set to `out_features` if not None (Default: None)
        - `dropout_p` (float): Alpha dropout probability
        """
        super().__init__()
        hidden_features = out_features if hidden_features is None else hidden_features

        self.snn_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_p),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_p),
        )

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.snn_block(x)


class BatchNormReLUBlock(nn.Module):
    """
    (Linear -> BatchNorm -> ReLU -> Dropout) * 2
    """

    def __init__(
        self, in_features: int, out_features: int, hidden_features: Optional[int] = None, dropout_p=0.0
    ) -> None:
        """
        (Linear -> BatchNorm -> ReLU -> Dropout) * 2

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features
        - `hidden_features` (int): number of hidden features. Set to `out_features` if not None (Default: None)
        - `dropout_p` (float): Alpha dropout probability
        """
        super().__init__()
        hidden_features = out_features if hidden_features is None else hidden_features

        self.bnr_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=hidden_features, out_features=out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bnr_block(x)


class LayerNormGELUBlock(nn.Module):
    """
    (Linear -> LayerNorm -> GELU -> Dropout) * 2
    """

    def __init__(
        self, in_features: int, out_features: int, hidden_features: Optional[int] = None, dropout_p=0.0
    ) -> None:
        """
        (Linear -> LayerNorm -> GELU -> Dropout) * 2

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features
        - `hidden_features` (int): number of hidden features. Set to `out_features` if not None (Default: None)
        - `dropout_p` (float): Alpha dropout probability
        """
        super().__init__()
        hidden_features = out_features if hidden_features is None else hidden_features

        self.lng_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features, bias=False),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=hidden_features, out_features=out_features, bias=False),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lng_block(x)
