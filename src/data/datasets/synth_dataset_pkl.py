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

    def __init__(self, path_to_dataset: Union[str, Path], mmap: bool = True):
        """
        Dataset used to load audio embeddings (targets) and synth parameters
        (features) from .pkl files.

        Args
        - `path_to_dataset` (Union[str, Path]): path to the folder containing the pickled files
        - `mmap` (bool): whether the audio_embeddings.pkl and synth_params.pkl tensors should be mmaped
        rather than loading all the storages into memory. This can be advantageous for large datasets.
        (Default: True)
        """
        super().__init__()

        path_to_dataset = Path(path_to_dataset)
        if not path_to_dataset.is_dir():
            raise ValueError(f"{path_to_dataset} is not a directory.")

        self.path_to_dataset = path_to_dataset

        with open(self.path_to_dataset / "configs.pkl", "rb") as f:
            self.configs_dict = torch.load(f)

        with open(self.path_to_dataset / "synth_parameters_description.pkl", "rb") as f:
            self._synth_params_descr = torch.load(f)

        # whether or not to mmap the dataset (pickled torch tensors)
        self.is_mmap = mmap

        # load the dataset in __getitem__() to avoid unexpected high memory usage when num_workers>0
        self.audio_embeddings = None
        self.synth_params = None

    @property
    def audio_fe_name(self) -> str:
        return self.configs_dict["audio_fe"]

    @property
    def embedding_dim(self) -> int:
        # return self.audio_embeddings.shape[1]
        return self.configs_dict["num_outputs"]

    @property
    def name(self) -> str:
        return self.path_to_dataset.stem

    @property
    def synth_name(self) -> str:
        return self.configs_dict["synth"]

    @property
    def num_used_synth_params(self) -> int:
        # return self.synth_params.shape[1]
        return len(self._synth_params_descr)

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
        # return len(self.audio_embeddings)
        return self.configs_dict["dataset_size"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        # load and mmap the dataset during the first call of __getitem__
        if self.audio_embeddings is None or self.synth_params is None:
            self._load_dataset()

        return self.synth_params[index], self.audio_embeddings[index]

    def _load_dataset(self) -> None:
        self.audio_embeddings = torch.load(
            str(self.path_to_dataset / "audio_embeddings.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.synth_params = torch.load(
            str(self.path_to_dataset / "synth_params.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        assert len(self.audio_embeddings) == len(self.synth_params)


if __name__ == "__main__":
    import os
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader

    NUM_EPOCH = 5
    PATH_TO_DATASET = (
        Path(os.environ["PROJECT_ROOT"])
        / "data"
        / "datasets"
        / "tal_noisemaker_mn04_size=10240000_seed=500_pkl_train-v1"
    )

    dataset = SynthDatasetPkl(PATH_TO_DATASET)

    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    # add timer here
    start = timer()
    for e in range(NUM_EPOCH):
        print(f"Epoch {e}")
        for i, (params, audio) in enumerate(loader):
            if i % 1000 == 0:
                print(f"{i} batch generated")
    print(f"Total time: {timer() - start}. Approximate time per epoch: {(timer() - start) / NUM_EPOCH}")

    print("")
