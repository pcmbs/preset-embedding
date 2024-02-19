import pytest
import torch
from torch import nn

from models.preset.mlp_builder import mlp_snn_raw, mlp_relu_raw, mlp_relu_oh
from models.preset.block_layers import SelfNormalizingBlock, BatchNormReLUBlock
from utils.synth import PresetHelper


@pytest.fixture
def tal_preset_helper():
    """Return a PresetHelper instance for the TAL-NoiseMaker synthesizer"""
    params_to_exclude_str = (
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

    return PresetHelper("tal_noisemaker", params_to_exclude_str)


@pytest.mark.parametrize(
    "out_features, kwargs",
    [
        (768, {}),
        (768, {"hidden_features": 2048, "num_blocks": 3, "block_kwargs": {"dropout_p": 0.1}}),
    ],
)
def test_mlp_snn_raw(tal_preset_helper, out_features, kwargs):
    mlp = mlp_snn_raw(out_features=out_features, preset_helper=tal_preset_helper, **kwargs)

    # assert that the layers are correct
    assert isinstance(mlp.norm_pre, nn.Identity)
    assert isinstance(mlp.blocks[0], SelfNormalizingBlock)

    # assert that the number of blocks is correct
    assert len(mlp.blocks) == kwargs.get("num_blocks", 2)

    # assert that the dropout probability is correct
    assert mlp.blocks[0].snn_block[2].p == kwargs.get("block_kwargs", {}).get("dropout_p", 0.0)

    # assert that the input size is correct
    # TODO: mod with more advance embedding layers
    assert mlp.embedding_layer.embedding_dim == tal_preset_helper.num_used_params
    assert mlp.blocks[0].snn_block[0].in_features == tal_preset_helper.num_used_params

    # assert that the hidden size is correct
    assert mlp.out_layer.in_features == kwargs.get("hidden_features", 1024)
    assert mlp.blocks[0].snn_block[0].out_features == kwargs.get("hidden_features", 1024)

    # assert that the output size is correct
    assert mlp.out_layer.out_features == out_features


@pytest.mark.parametrize(
    "out_features, kwargs",
    [
        (768, {}),
        (768, {"hidden_features": 2048, "num_blocks": 3, "block_kwargs": {"dropout_p": 0.1}}),
    ],
)
def test_mlp_relu_raw(tal_preset_helper, out_features, kwargs):
    mlp = mlp_relu_raw(out_features=out_features, preset_helper=tal_preset_helper, **kwargs)

    # assert that the layers are correct
    assert isinstance(mlp.norm_pre, nn.Identity)
    assert isinstance(mlp.blocks[0], BatchNormReLUBlock)

    # assert that the number of blocks is correct
    assert len(mlp.blocks) == kwargs.get("num_blocks", 2)

    # assert that the dropout probability is correct
    assert mlp.blocks[0].bnr_block[3].p == kwargs.get("block_kwargs", {}).get("dropout_p", 0.0)

    # assert that the input size is correct
    # TODO: mod with more advance embedding layers
    assert mlp.embedding_layer.embedding_dim == tal_preset_helper.num_used_params
    assert mlp.blocks[0].bnr_block[0].in_features == tal_preset_helper.num_used_params

    # assert that the hidden size is correct
    assert mlp.out_layer.in_features == kwargs.get("hidden_features", 1024)
    assert mlp.blocks[0].bnr_block[0].out_features == kwargs.get("hidden_features", 1024)

    # assert that the output size is correct
    assert mlp.out_layer.out_features == out_features


@pytest.mark.parametrize(
    "out_features, kwargs",
    [
        (768, {}),
        (768, {"hidden_features": 2048, "num_blocks": 3, "block_kwargs": {"dropout_p": 0.1}}),
    ],
)
def test_mlp_relu_oh(tal_preset_helper, out_features, kwargs):
    mlp = mlp_relu_oh(out_features=out_features, preset_helper=tal_preset_helper, **kwargs)

    # assert that the layers are correct
    assert isinstance(mlp.norm_pre, nn.Identity)
    assert isinstance(mlp.blocks[0], BatchNormReLUBlock)

    # assert that the number of blocks is correct
    assert len(mlp.blocks) == kwargs.get("num_blocks", 2)

    # assert that the dropout probability is correct
    assert mlp.blocks[0].bnr_block[3].p == kwargs.get("block_kwargs", {}).get("dropout_p", 0.0)

    # assert that the input size is correct
    assert mlp.blocks[0].bnr_block[0].in_features == mlp.embedding_layer.embedding_dim

    # assert that the hidden size is correct
    assert mlp.out_layer.in_features == kwargs.get("hidden_features", 1024)
    assert mlp.blocks[0].bnr_block[0].out_features == kwargs.get("hidden_features", 1024)

    # assert that the output size is correct
    assert mlp.out_layer.out_features == out_features
