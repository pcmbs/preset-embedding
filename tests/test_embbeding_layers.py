# pylint: disable=E1102
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets import SynthDataset
from models.preset.embedding_layers import RawParameters, OneHotEncoding
from utils.synth import PresetHelper

NUM_SAMPLES = 32


@pytest.fixture
def tal_dataset():
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

    p_helper = PresetHelper("tal_noisemaker", params_to_exclude_str)

    dataset = SynthDataset(p_helper, NUM_SAMPLES, seed_offset=5423)

    return dataset


def test_raw_params_emb_layer(tal_dataset):
    """
    Test that the RawParameters embedding layer returns the correct raw parameter values
    given the index of a category from a categorical synth parameters
    """

    loader = DataLoader(tal_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = RawParameters(preset_helper=tal_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    raw_params_emb = emb_layer(params.clone())
    # TODO: refactor that since fn doesnt exist anymore
    for (cat_values, _), indices in tal_dataset.preset_helper.grouped_used_params["discrete"]["cat"].items():
        for i in indices:
            for sample, emb in zip(params, raw_params_emb):
                assert cat_values[int(sample[i])] == emb[i]


def test_raw_params_emb_layer_out_shape(tal_dataset):
    """
    Test that the RawParameters embedding layer returns the correct shape
    """

    loader = DataLoader(tal_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = RawParameters(preset_helper=tal_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    raw_params_emb = emb_layer(params.clone())

    assert len(raw_params_emb) == len(params)

    for sample, emb in zip(params, raw_params_emb):
        assert emb.shape == sample.shape
        assert emb.shape[0] == tal_dataset.num_used_parameters


def test_onehot_params_emb_layer(tal_dataset):
    """
    Test that the OneHotEncoding correctly embed the categorical parameters
    """

    loader = DataLoader(tal_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = OneHotEncoding(preset_helper=tal_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    onehot_params_emb = emb_layer(params.clone())

    offset = emb_layer.num_non_cat_params

    for (cat_values, _), indices in tal_dataset.preset_helper.grouped_used_params["discrete"]["cat"].items():
        cat_card = len(cat_values)
        for i in indices:
            for sample, emb in zip(params, onehot_params_emb):
                assert torch.all(
                    F.one_hot(sample[i].to(torch.long), cat_card) == emb[offset : offset + cat_card]
                )
            offset += cat_card
