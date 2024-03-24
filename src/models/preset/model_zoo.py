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


##############################################################################################################
#### MLP based models
##############################################################################################################
def mlp_raw(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks and and raw parameter values in range [0,1].
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=MLPBlock,
        hidden_features=kwargs.get("hidden_features", 2560),
        num_blocks=kwargs.get("num_blocks", 1),
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
        hidden_features=kwargs.get("hidden_features", 2560),
        num_blocks=kwargs.get("num_blocks", 1),
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
        hidden_features=kwargs.get("hidden_features", 768),
        num_blocks=kwargs.get("num_blocks", 6),
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


def highway_ft(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    Highway MLP with flattened features tokenization embedding.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizer,
        block_layer=HighwayBlock,
        hidden_features=kwargs.get("hidden_features", 512),
        num_blocks=kwargs.get("num_blocks", 6),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.0}),
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": kwargs.get("token_dim", 64),
            "pe_type": None,
            "has_cls": False,
            "pe_dropout_p": kwargs.get("pe_dropout_p", 0.0),
        },
    )


def highway_ftgru(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    Highway MLP with PresetTokenizer+GRU embedding.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizerWithGRU,
        block_layer=HighwayBlock,
        hidden_features=kwargs.get("hidden_features", 768),
        num_blocks=kwargs.get("num_blocks", 6),
        block_kwargs=kwargs.get("block_kwargs", {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.0}),
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": kwargs.get("token_dim", 384),
            "pre_norm": kwargs.get("pre_norm", False),
            "gru_hidden_factor": kwargs.get("gru_hidden_factor", 1),
            "gru_num_layers": kwargs.get("gru_num_layers", 1),
            "gru_dropout_p": kwargs.get("gru_dropout_p", 0.0),
            "pe_dropout_p": kwargs.get("pe_dropout_p", 0.0),
        },
    )


##############################################################################################################
#### GRU based models
##############################################################################################################
def gru_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    Bi-GRU with One-Hot encoded categorical synthesizer parameters.
    """
    return GRUBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_layers=kwargs.get("num_layers", 2),
        dropout_p=kwargs.get("dropout_p", 0.0),
        embedding_kwargs={"preset_helper": preset_helper},
    )


##############################################################################################################
#### TFM based models
##############################################################################################################
def tfm(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    TODO
    """
    return TfmBuilder(
        out_features=out_features,
        tokenizer=PresetTokenizer,
        pe_type=kwargs.get("pe_type", "absolute"),
        hidden_features=kwargs.get("hidden_features", 256),
        num_blocks=kwargs.get("num_blocks", 6),
        num_heads=kwargs.get("num_heads", 4),
        mlp_factor=kwargs.get("mlp_factor", 2.0),
        pooling_type=kwargs.get("pooling_type", "cls"),
        last_activation=kwargs.get("last_activation", "ReLU"),
        tokenizer_kwargs={"preset_helper": preset_helper, "pe_dropout_p": kwargs.get("pe_dropout_p", 0.0)},
        block_kwargs=kwargs.get("block_kwargs", {"activation": "relu", "dropout": 0.0}),
    )


if __name__ == "__main__":

    print("")
