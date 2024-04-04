from torch import nn

from models.preset.embedding_layers import (
    OneHotEncoding,
    PresetTokenizer,
    PresetTokenizerWithGRU,
    RawParameters,
)
from models.preset.gru_builder import GRUBuilder
from models.preset.mlp_builder import MlpBuilder, MLPBlock, HighwayBlock, ResNetBlock
from models.preset.tfm_builder import TfmBuilder
from utils.synth import PresetHelper

# TODO: Documentation
# TODO: Add support to load weights from a checkpoint (will need to extract them from lightning ckpt first)


##############################################################################################################
#### MLP based models
##############################################################################################################
def mlp_raw(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 1,
    hidden_features: int = 2048,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks (by default) and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=MLPBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 1,
    hidden_features: int = 2048,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP with One-Hot encoded categorical synthesizer parameters.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=MLPBlock,
        num_blocks=num_blocks,
        hidden_features=hidden_features,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def highway_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 768,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    Highway MLP and One-Hot encoded categorical synthesizer parameters.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def resnet_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 4,
    hidden_features: int = 256,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
    block_residual_dropout_p: float = 0.0,
) -> nn.Module:
    """
    ResNet with One-Hot encoded categorical synthesizer parameters.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=ResNetBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={
            "norm": block_norm,
            "act_fn": block_act_fn,
            "dropout_p": block_dropout_p,
            "residual_dropout_p": block_residual_dropout_p,
        },
        embedding_kwargs={"preset_helper": preset_helper},
    )


def highway_ft(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 512,
    token_dim: int = 64,
    pe_dropout_p: float = 0.0,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    Highway MLP with flattened features tokenization embedding.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizer,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": token_dim,
            "pe_type": None,
            "has_cls": False,
            "pe_dropout_p": pe_dropout_p,
        },
    )


def highway_ftgru(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 768,
    token_dim: int = 384,
    pe_dropout_p: float = 0.0,
    gru_hidden_factor: float = 1.0,
    gru_num_layers: int = 1,
    gru_dropout_p: float = 0.0,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
    pre_norm: bool = False,
) -> nn.Module:
    """
    Highway MLP with PresetTokenizer+GRU embedding.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizerWithGRU,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": token_dim,
            "pre_norm": pre_norm,
            "gru_hidden_factor": gru_hidden_factor,
            "gru_num_layers": gru_num_layers,
            "gru_dropout_p": gru_dropout_p,
            "pe_dropout_p": pe_dropout_p,
        },
    )


##############################################################################################################
#### GRU based models
##############################################################################################################
def gru_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_layers: int = 1,
    hidden_features: int = 1024,
    dropout_p: float = 0.0,
) -> nn.Module:
    """
    Bi-GRU with One-Hot encoded categorical synthesizer parameters.
    """
    return GRUBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout_p=dropout_p,
        embedding_kwargs={"preset_helper": preset_helper},
    )


##############################################################################################################
#### TFM based models
##############################################################################################################
def tfm(
    out_features: int,
    preset_helper: PresetHelper,
    pe_type: str = "absolute",
    num_blocks: int = 6,
    hidden_features: int = 256,
    num_heads: int = 8,
    mlp_factor: float = 4.0,
    pooling_type: str = "cls",
    last_activation: str = "ReLU",
    pe_dropout_p: float = 0.0,
    block_activation: str = "relu",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """ """
    return TfmBuilder(
        out_features=out_features,
        tokenizer=PresetTokenizer,
        pe_type=pe_type,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_factor=mlp_factor,
        pooling_type=pooling_type,
        last_activation=last_activation,
        tokenizer_kwargs={"preset_helper": preset_helper, "pe_dropout_p": pe_dropout_p},
        block_kwargs={"activation": block_activation, "dropout": block_dropout_p},
    )


if __name__ == "__main__":

    print("")
