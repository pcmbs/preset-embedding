"""
Module implementing various MLP based preset encoder.
"""

from typing import Dict, Optional

import torch
from torch import nn

from models.preset.embedding_layers import RawParameters, OneHotEncoding, FTTokenizer, PositionalEncoding
from utils.synth import PresetHelper

#################### BUILDING BLOCKS ####################


class MLPBlock(nn.Module):
    """
    (Linear -> norm -> act_fn -> Dropout) * 2
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        norm: str = "BatchNorm1d",
        act_fn: str = "ReLU",
        dropout_p=0.0,
    ) -> None:
        """
        (Linear -> norm -> act_fn -> Dropout) * 2

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features
        - `hidden_features` (int): number of hidden features. Set to `out_features` if not None (Default: None)
        - `norm` (nn.Module): normalization layer. Must be nn.LayerNorm or nn.BatchNorm1d
        - `act_fn` (nn.Module): activation function
        - `dropout_p` (float): dropout probability
        """
        super().__init__()
        hidden_features = out_features if hidden_features is None else hidden_features

        self.act_fn = getattr(nn, act_fn)
        if norm not in ["LayerNorm", "BatchNorm1d"]:
            raise ValueError(f"norm must be 'LayerNorm' or 'BatchNorm1d', got {norm}")
        self.norm = getattr(nn, norm)

        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features, bias=False),
            self.norm(hidden_features),
            self.act_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=hidden_features, out_features=out_features, bias=False),
            self.norm(out_features),
            self.act_fn(),
            nn.Dropout(p=dropout_p),
        )

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetBlock(nn.Module):
    """
    MLP based ResNet adapted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    ResNetBlock(x) = ReLU(x + residual_dropout(norm(linear(dropout(act_fn(norm(linear(x)))))))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: str = "BatchNorm1d",
        act_fn: str = "ReLU",
        dropout_p=0.0,
        residual_dropout_p=0.0,
        has_residual: bool = True,
    ) -> None:
        """
        MLP based ResNet adapted from
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

        ResNetBlock(x) = ReLU(x + residual_dropout(norm(linear(dropout(act_fn(norm(linear(x)))))))

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features.
        If different from `in_features` the residual connection is removed,
        such that the output is act_fn(norm(nn.Linear(x))).
        - `norm` (nn.Module): normalization layer. Must be nn.LayerNorm or nn.BatchNorm1d
        - `act_fn` (nn.Module): activation function
        - `dropout_p` (float): dropout probability
        - `residual_dropout_p` (float): dropout probability for the residual connection.
        - `has_residual` (bool): whether to include the residual connection or not.
        Note that the residual connection is removed if `in_features` is not equal to `out_features`.
        """
        super().__init__()
        self.has_residual = has_residual and in_features == out_features

        if norm not in ["LayerNorm", "BatchNorm1d"]:
            raise ValueError(f"norm must be 'LayerNorm' or 'BatchNorm1d', got {norm}")
        self.norm = getattr(nn, norm)
        self.act_fn = getattr(nn, act_fn)

        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            self.norm(out_features),
            self.act_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=out_features, out_features=out_features, bias=False),
            self.norm(out_features),
        )
        self.residual_dropout = nn.Dropout(p=residual_dropout_p)
        self.last_activation = self.act_fn()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        if self.has_residual:
            x = identity + self.residual_dropout(x)
        x = self.last_activation(x)
        return x


class HighwayBlock(nn.Module):
    """
    Highway MLP Block as proposed by
    Srivastava, R.K., Greff, K. and Schmidhuber, J. (2015) 'Highway Networks'.
    Available at: http://arxiv.org/abs/1505.00387

    HighwayBlock(x) = fc(x) * gate(x) + x * (1 - gate(x))
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        norm: str = "BatchNorm1d",
        act_fn: str = "ReLU",
        dropout_p: float = 0.0,
        residual_dropout_p: float = 0.0,
        has_residual: bool = True,
    ) -> None:
        """
        Highway MLP Block as proposed by
        Srivastava, R.K., Greff, K. and Schmidhuber, J. (2015) 'Highway Networks'.
        Available at: http://arxiv.org/abs/1505.00387

        HighwayBlock(x) = fc(x) * gate(x) + x * (1 - gate(x))

        Args
        - `in_features` (int): number of input features
        - `out_features` (int): number of output features.
        If different from `in_features` the residual connection is removed,
        such that the output is act_fn(norm(nn.Linear(x))).
        - `norm` (nn.Module): normalization layer. Must be nn.LayerNorm or nn.BatchNorm1d
        - `act_fn` (nn.Module): activation function
        - `dropout_p` (float): dropout probability
        - `has_residual` (bool): whether to include the residual connection or not.
        Note that the residual connection is removed if `in_features` is not equal to `out_features`.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_residual = has_residual and self.in_features == self.out_features

        self.act_fn = getattr(nn, act_fn)
        if norm not in ["LayerNorm", "BatchNorm1d"]:
            raise ValueError(f"norm must be 'LayerNorm' or 'BatchNorm1d', got {norm}")
        self.norm = getattr(nn, norm)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False),
            self.norm(self.out_features),
            self.act_fn(),
            nn.Dropout(p=dropout_p),
        )

        if self.has_residual:
            self.gate = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=True),
                nn.Sigmoid(),
                nn.Dropout(p=residual_dropout_p),
            )

        self.init_weights()

    def init_weights(self) -> None:
        # FC layer (linear and normalization layers)
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
        block_kwargs: Optional[Dict] = None,
        embedding_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer(**embedding_kwargs)
        self.in_features = self.embedding_layer.embedding_dim * self.embedding_layer.out_length

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

        self.head = nn.Linear(hidden_features, out_features)

        self.init_weights()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        self.embedding_layer.init_weights()

        for b in self.blocks:
            b.init_weights()

        nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        x = self.blocks(x)
        x = self.head(x)
        return x


#################### MODELS ####################


def mlp_raw(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks and and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=MLPBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=MLPBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def highway_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    Highway MLP and One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=HighwayBlock,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_blocks=kwargs.get("num_blocks", 2),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.0}),
        embedding_kwargs={"preset_helper": preset_helper},
    )


def resnet_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    ResNet with One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=ResNetBlock,
        hidden_features=kwargs.get("hidden_features", 256),
        num_blocks=kwargs.get("num_blocks", 4),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU"}),
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

    # model = resnet_oh(192, p_helper, num_blocks=6, hidden_features=512)
    # model = highway_oh(192, p_helper, num_blocks=12, hidden_features=512)
    # print(model.num_parameters)
    # print(model)
    model = mlp_oh  # resnet_oh
    print("[num_blocks, hidden_features] -> number of parameters")
    for i in range(1, 4):
        for j in range(128, 4096 + 128, 128):
            net = model(192, p_helper, num_blocks=i, hidden_features=j)
            print(f"[{i:>2}, {j:>4}] -> {net.num_parameters:.3e}")
    print("")
