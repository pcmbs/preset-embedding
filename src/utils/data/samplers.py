"""Module implementing a custom torch SequentialSampler allowing to resume training from a given sample index."""
from typing import Iterator, Sized
from torch.utils.data import Sampler


class SequentialSampler2(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, start_idx: int = 0) -> None:
        super().__init__()
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source) - self.start_idx
