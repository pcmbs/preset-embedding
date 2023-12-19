"""
Module implementing a torch Dataset used to load audio embeddings (targets) and
synth parameters (features) from .pkl files.
"""
from pathlib import Path
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset


class SynthDatasetPkl(Dataset):
    """
    Dataset used to load audio embeddings (targets) and synth parameters
    (features) from .pkl files.
    """

    def __init__(self, path_to_dataset: Union[str, Path]):
        super().__init__()

        path_to_dataset = Path(path_to_dataset) if isinstance(path_to_dataset, str) else path_to_dataset
        if not path_to_dataset.is_dir():
            raise ValueError(f"{path_to_dataset} is not a directory.")

        self.path_to_dataset = path_to_dataset

        with open(self.path_to_dataset / "configs.pkl", "rb") as f:
            self.configs_dict = torch.load(f)

        with open(self.path_to_dataset / "synth_parameters_description.pkl", "rb") as f:
            self._synth_params_descr = torch.load(f)

        with open(self.path_to_dataset / "synth_params.pkl", "rb") as f:
            self.synth_params = torch.load(f)

        with open(self.path_to_dataset / "audio_embeddings.pkl", "rb") as f:
            self.audio_embeddings = torch.load(f)

        assert len(self.audio_embeddings) == len(self.synth_params)

    @property
    def audio_fe_name(self) -> str:
        return self.configs_dict["audio_fe"]

    @property
    def embedding_dim(self) -> int:
        return self.audio_embeddings.shape[1]

    @property
    def synth_name(self) -> str:
        return self.configs_dict["synth"]

    @property
    def num_used_synth_params(self) -> int:
        return self.synth_params.shape[1]

    @property
    def synth_parameters_description(self) -> List[Tuple[int, str, int]]:
        """Return the description of the used synthesizer parameters as a
        list of tuple (feature_idx, synth_param_name, synth_param_idx)."""
        return self._synth_params_descr

    @property
    def embedding_size_in_mb(self) -> float:
        return round(self.audio_embeddings.element_size() * self.audio_embeddings.nelement() * 1e-6, 2)

    @property
    def synth_params_size_in_mb(self) -> float:
        return round(self.synth_params.element_size() * self.synth_params.nelement() * 1e-6, 2)

    def __len__(self) -> int:
        return len(self.audio_embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.synth_params[index], self.audio_embeddings[index]


if __name__ == "__main__":
    import os

    PATH_TO_DATASET = (
        Path(os.environ["PROJECT_ROOT"])
        / "div_check"
        / "data"
        / "tal_noisemaker_mn04_size=65536_seed=45858_2023-12-13_14-19-35"
    )

    dataset = SynthDatasetPkl(PATH_TO_DATASET)

    print("")
