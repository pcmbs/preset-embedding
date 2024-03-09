"""
Module implementing various MLP based preset encoder.
"""

from typing import Dict, Optional

import torch
from torch import nn

from models.preset.embedding_layers import RawParameters, OneHotEncoding
from utils.synth import PresetHelper

#################### BUILDING BLOCKS ####################


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
        - `dropout_p` (float): dropout probability
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
        - `dropout_p` (float): dropout probability
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


class HighwayBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        dropout_p: float = 0.0,
    ) -> None:
        """
        Highway MLP Block with LayerNorm and GELU activation.

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features.
        If different from `in_features` the residual connection and gates are removed,
        uch that the output is GELU(LayerNorm(FC(x))).
        - `dropout_p` (float): dropout probability
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_residual = self.in_features == self.out_features
        self.dropout_p = dropout_p

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False),
            nn.LayerNorm(self.out_features),
            nn.GELU(),
        )

        if self.has_residual:
            self.gate = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
                nn.Sigmoid(),
                nn.Dropout(p=self.dropout_p),
            )

        self.init_weights()

    def init_weights(self) -> None:
        # FC layer (linear layer and layer norm)
        nn.init.kaiming_normal_(self.fc[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.ones_(self.fc[1].weight)
        nn.init.zeros_(self.fc[1].bias)

        # Gate layer
        if self.has_residual:
            nn.init.xavier_normal_(self.gate[0].weight)
            nn.init.constant_(self.gate[0].bias, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Highway MLP layer.
        """
        if not self.has_residual:
            return self.fc(x)

        fc_out = self.fc(x)
        gate_out = self.gate(x)
        return fc_out * gate_out + x * (1 - gate_out)


#################### MLP BUILDER ####################


class MlpBuilder(nn.Module):
    def __init__(
        self,
        out_features: int,
        embedding_layer: nn.Module,
        block_layer: nn.Module,
        hidden_features: int = 1024,
        num_blocks: int = 2,
        pre_norm: bool = False,
        block_kwargs: Optional[Dict] = None,
        embedding_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer(**embedding_kwargs)
        self.in_features = self.embedding_layer.embedding_dim * self.embedding_layer.out_length

        self.norm_pre = nn.LayerNorm(self.in_features) if pre_norm else nn.Identity()

        self.blocks = nn.Sequential(
            *[
                block_layer(
                    in_features=self.in_features if b == 0 else hidden_features,
                    out_features=hidden_features,
                    **block_kwargs,
                )
                for b in range(num_blocks)
            ]
        )

        self.out_layer = nn.Linear(hidden_features, out_features)

        self.init_weights()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        self.embedding_layer.init_weights()

        if isinstance(self.norm_pre, nn.LayerNorm):
            nn.init.ones_(self.norm_pre.weight)
            nn.init.zeros_(self.norm_pre.bias)

        for b in self.blocks:
            b.init_weights()

        nn.init.kaiming_normal_(self.out_layer.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.out_layer(x)
        return x


#################### MODELS ####################


def mlp_snn_raw(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with Self-Normalizing Blocks and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=SelfNormalizingBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_relu_raw(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks and and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=BatchNormReLUBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_relu_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks and and One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=BatchNormReLUBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_gelu_raw(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with LayerNorm+GeLU blocks and and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=LayerNormGELUBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_gelu_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with LayerNorm+GeLU blocks and and One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=LayerNormGELUBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_hw_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    Highway MLP with LayerNorm+ReLU activation and One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=HighwayBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


if __name__ == "__main__":

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
    # model = mlp_gelu_oh(192, p_helper)
    model = mlp_hw_oh(192, p_helper, num_blocks=12, hidden_features=1024)
    print(model)
    print(model.num_parameters)
    print("")
