import math
from typing import Dict, Tuple
import torch
from torch import nn

from utils.synth import PresetHelper

# https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py


class Tokenizer(nn.Module):
    """
    Synth Presets Tokenizer class.

    - Each non-categorical parameter is embedded using a distinct linear projection.
    - Each categorical parameter is embedded using a nn.Embedding lookup table.
    """

    def __init__(self, preset_helper: PresetHelper, embedding_dim: int) -> None:
        """
        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.
        - `embedding_dim` (int): The embedding dimension.
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._out_length = preset_helper.num_used_parameters

        self.noncat_idx = torch.tensor(preset_helper.used_noncat_parameters_idx, dtype=torch.long)
        self.noncat_tokenizer = nn.Parameter(torch.zeros(len(self.noncat_idx), embedding_dim))

        self.cat_idx = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        self.cat_offsets, total_num_cat = self._compute_cat_infos(preset_helper)
        self.cat_tokenizer = nn.Embedding(num_embeddings=total_num_cat, embedding_dim=embedding_dim)

        # if pos_enc == "sin_cos":
        #     pass
        # elif pos_enc == "learned":
        #     self.pos_enc = nn.Parameter(torch.zeros(self.out_length, embedding_dim))
        # elif pos_enc == "bias":
        #     self.pos_enc = nn.Parameter(torch.zeros(self.out_length, 1))

        self.init_weights()

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def init_weights(self):
        nn.init.normal_(self.noncat_tokenizer, std=0.02)
        nn.init.normal_(self.cat_tokenizer.weight, std=0.02)

    def _compute_cat_infos(self, preset_helper: PresetHelper) -> Tuple[torch.Tensor, int]:
        """
        Compute the offsets for each categorical parameter and the total number of categories
        (i.e., sum over all categorical parameters' cardinality).

        Args
        - `preset_helper` (PresetHelper): An instance of PresetHelper for a given synthesizer.

        Returns
        - `cat_offsets` (torch.Tensor): the offsets for each categorical parameter as a list cat_offsets[cat_param_idx] = offset.
        - `total_num_cat` (int):  total number of categories.
        """
        cat_offsets = []
        offset = 0
        for (cat_values, _), indices in preset_helper.grouped_used_parameters["discrete"]["cat"].items():
            for _ in indices:
                cat_offsets.append(offset)
                offset += len(cat_values)
        total_num_cat = offset
        cat_offsets = torch.tensor(cat_offsets, dtype=torch.long)
        return cat_offsets, total_num_cat

    def forward(self, x):
        tokens = torch.zeros((*x.shape, self._embedding_dim), device=x.device)

        # Assign noncat embeddings
        noncat_tokens = self.noncat_tokenizer * x[:, self.noncat_idx, None]
        tokens[:, self.noncat_idx] = noncat_tokens

        # Assign cat embeddings
        cat_tokens = x[:, self.cat_idx].to(dtype=torch.long) + self.cat_offsets
        tokens[:, self.cat_idx] = self.cat_tokenizer(cat_tokens)

        return tokens


# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# class PositionalEncoding(nn.Module):

#     def __init__(self, embedding_dim: int, num_tokens: int, max_len: int = 5000):
#         super().__init__()

#         position = torch.arange(num_tokens).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
#         pe = torch.zeros(num_tokens, 1, embedding_dim)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         return x + self.pe[: x.size(0)]


if __name__ == "__main__":
    import os
    from pathlib import Path
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader
    from data.datasets import SynthDatasetPkl

    DATASET_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"
    DATASET_PATH = DATASET_FOLDER / "talnm_mn04_size=65536_seed=45858_dev_val_v1"

    BATCH_SIZE = 512
    EMB_DIM = 128

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

    dataset = SynthDatasetPkl(path_to_dataset=DATASET_PATH, mmap=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    tokenizer = Tokenizer(p_helper, EMB_DIM)

    start = timer()
    for synth_params, _ in loader:
        tokens = tokenizer(synth_params)
    print(f"Total time: {timer() - start}")
    print(f"Tokenizer parameters: {sum(p.numel() for p in tokenizer.parameters())}")
    print("")
