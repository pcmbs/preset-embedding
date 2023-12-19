from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.functional import resample


class SynthDatasetWav(Dataset):
    def __init__(self, path_to_dataset: Union[Path, str], sample_rate: float):
        path_to_dataset = Path(path_to_dataset) if isinstance(path_to_dataset, str) else path_to_dataset
        if not path_to_dataset.is_dir():
            raise ValueError(f"{path_to_dataset} is not a directory.")

        self._path_to_dataset = path_to_dataset

        self._file_stems = sorted(
            [p.stem for p in self._path_to_dataset.glob("*.wav")],
            key=lambda i: int(i.split("_")[-1]),
        )

        self._sample_rate = sample_rate

        with open(self._path_to_dataset / f"{self._file_stems[0]}.wav", "rb") as f:
            tmp_audio, tmp_sample_rate = torchaudio.load(f)
        self._channels = tmp_audio.shape[0]
        self._audio_length_sec = tmp_audio.shape[1] // tmp_sample_rate

        with open(self._path_to_dataset / f"{self._file_stems[0]}.pkl", "rb") as f:
            tmp_params = torch.load(f)
        self._num_used_params = tmp_params.shape[0]

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_used_parameters(self):
        return self._num_used_params

    @property
    def audio_length_sec(self):
        return self._audio_length_sec

    @property
    def channels(self):
        return self._channels

    @property
    def path_to_audio(self):
        return self._path_to_dataset

    @property
    def file_stems(self):
        return self._file_stems

    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        name = self.file_stems[index]

        with open(self._path_to_dataset / f"{name}.wav", "rb") as f:
            audio, data_sample_rate = torchaudio.load(f)

        if data_sample_rate != self.sample_rate:
            audio = audio.clone()  # might raise an error without
            audio = resample(audio, data_sample_rate, self.sample_rate)

        with open(self._path_to_dataset / f"{name}.pkl", "rb") as f:
            params = torch.load(f)

        _ = 0
        return _, params, audio, _


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    dataset = SynthDatasetWav(
        path_to_dataset=Path(os.environ["PROJECT_ROOT"])
        / "div_check"
        / "data"
        / "NoiseMakerDataset_small_231118",
        sample_rate=32_000,
    )

    loader = DataLoader(dataset, batch_size=None, shuffle=False)

    for i, (_, params, audio, _) in enumerate(loader):
        if i == 10:
            break
        print("")

    print(dataset)
