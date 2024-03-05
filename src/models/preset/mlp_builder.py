from typing import Dict, Optional

import torch
from torch import nn
from models.preset.block_layers import SelfNormalizingBlock, BatchNormReLUBlock, LayerNormGELUBlock
from models.preset.embedding_layers import RawParameters, OneHotEncoding
from utils.synth import PresetHelper

########## MLP Builder Class Definition


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
        self.in_features = self.embedding_layer.embedding_dim

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
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.out_layer(x)
        return x


########## MLP Builder Functions


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
    MLP with LazerNorm+GeLU blocks and and raw parameter values in range [0,1].
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

    p_helper = PresetHelper("tal_noisemaker", PARAMETERS_TO_EXCLUDE_STR)
    model = mlp_gelu_oh(192, p_helper)
    print(model)
    print(sum(p.numel() for p in model.parameters()))
    print("")
