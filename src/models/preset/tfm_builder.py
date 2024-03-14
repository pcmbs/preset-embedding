import math
from typing import Dict
import torch
from torch import nn

from models.preset.embedding_layers import FTTokenizer
from utils.synth import PresetHelper

# Resources:
# https://github.com/gwendal-lv/spinvae2/blob/main/model/presetencoder.py
# https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py


class TfmBuilder(nn.Module):
    def __init__(
        self,
        out_features: int,
        # Tokenizer
        tokenizer: nn.Module,
        pe_type: str,
        # Encoder
        hidden_features: int,
        num_blocks: int,
        num_heads: int,
        mlp_factor: float,
        pooling_type: str,
        # Head
        last_activation: str,
        # Kwargs
        tokenizer_kwargs: Dict,
        block_kwargs: Dict,
    ) -> None:
        super().__init__()
        assert pooling_type in ["avg", "cls"], "pooling_type should be 'avg' or 'cls'"
        self.pooling_type = pooling_type

        self.tokenizer = tokenizer(
            embedding_dim=hidden_features, has_cls=pooling_type == "cls", pe_type=pe_type, **tokenizer_kwargs
        )

        # Encoder
        assert hidden_features % num_heads == 0, "embedding_dim should be divisible by num_heads"
        dim_feedforward = int(hidden_features * mlp_factor)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            **block_kwargs,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_blocks,
            enable_nested_tensor=not encoder_layer.norm_first,
        )

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_features),
            getattr(nn, last_activation)(),
            nn.Linear(hidden_features, out_features=out_features),
        )

        self.init_weights()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        self.tokenizer.init_weights()

        for name, param in self.encoder.layers.named_parameters():
            if "linear" in name or "self_attn" in name:
                if "weight" in name:
                    nn.init.trunc_normal_(param, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

            if "norm" in name:
                if "weight" in name:
                    nn.init.ones_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.encoder(x)
        # reduce tokens dimension before last layer (avg pooling or CLS token)
        # (batch_size, num_tokens, embedding_dim) -> (batch_size, embedding_dim)
        x = x[:, 1:].mean(-2) if self.pooling_type == "avg" else x[:, 0]
        x = self.head(x)
        return x


def tfm_base(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    TODO
    """
    return TfmBuilder(
        out_features=out_features,
        tokenizer=FTTokenizer,
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
    import os
    from pathlib import Path
    from torch.utils.data import DataLoader
    from data.datasets import SynthDatasetPkl

    SYNTH = "diva"
    BATCH_SIZE = 32
    OUT_FEATURES = 192

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"

    if SYNTH == "talnm":
        DATASET_PATH = DATASET_FOLDER / "talnm_mn04_size=65536_seed=45858_dev_val_v1"
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

    if SYNTH == "diva":
        DATASET_PATH = DATASET_FOLDER / "diva_mn04_size=65536_seed=400_hpo_val_v1"
        PARAMETERS_TO_EXCLUDE_STR = (
            "main:output",
            "vcc:*",
            "opt:*",
            "scope1:*",
            "clk:*",
            "arp:*",
            "plate1:*",
            "delay1:*",
            "chrs2:*",
            "phase2:*",
            "rtary2:*",
            "*keyfollow",
            "*velocity",
            "env1:model",
            "env2:model",
            "*trigger",
            "*release_on",
            "env1:quantise",
            "env2:quantise",
            "env1:curve",
            "env2:curve",
            "lfo1:sync",
            "lfo2:sync",
            "lfo1:restart",
            "lfo2:restart",
            "mod:rectifysource",
            "mod:invertsource",
            "mod:addsource*",
            "*revision",
            "vca:pan",
            "vca:volume",
            "vca:vca",
            "vca:panmodulation",
            "vca:panmoddepth",
            "vca:mode",
            "vca:offset",
        )

    p_helper = PresetHelper(SYNTH, PARAMETERS_TO_EXCLUDE_STR)

    dataset = SynthDatasetPkl(path_to_dataset=DATASET_PATH, mmap=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[num_blocks, hidden_features, mlp_factor] -> number of parameters")
    for i in [1, 2, 3, 4, 5, 6]:  # num_blocks
        for j in [128, 256, 512, 768]:  # embedding_dim
            for k in [1, 2, 3, 4]:  # mlp_ratio
                tfm = tfm_base(OUT_FEATURES, p_helper, num_blocks=i, hidden_features=j, mlp_factor=k)
                print(f"[{i}, {j}, {k}] -> {sum(p.numel() for p in tfm.encoder.parameters()):>9}")

    # tfm = tfm_base(192, p_helper, num_blocks=2, embedding_dim=256, mlp_ratio=2)
    # tfm.to(DEVICE)
    # print(f"Total number of parameters: {tfm.num_parameters}")
    # print(f"-> Tokenizer: {tfm.tokenizer.num_parameters}")
    # print(f"-> TFM: {sum(p.numel() for p in tfm.encoder.parameters())}")
    # print(tfm)

    # for synth_params, _ in loader:
    #     out = tfm(synth_params.to(DEVICE))
    #     break
    # print("")
