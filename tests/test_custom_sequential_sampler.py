"""
Test the integration of a custom torch SequentialSampler in a lighthning module which can resume training
from a given iteration number.

When using multiple workers for dataloading, the custom SequentialSampler needs to be instantiated before
being passed to the lightning module, otherwise only worker 0 will get the corrent start idx 
(the other ones taking the values of 0 (default)).
"""

from typing import Any, Dict

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning import LightningModule, seed_everything, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

from utils.data import SequentialSampler2

DATASET_SIZE = 100_000_000_000
SEED_OFFSET = 100
NUM_OUT_FEATURES = 69
BATCH_SIZE = 32
MAX_STEPS = 10
SAMPLER_LAST_IDX_TARGET = BATCH_SIZE * MAX_STEPS - 1


class DummyDataset(Dataset):
    def __init__(self, dataset_size, seed_offset=0, num_out_features=69):
        super().__init__()
        self.dataset_size = dataset_size
        self.seed_offset = seed_offset
        self.num_out_features = num_out_features

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        rng_cpu = torch.Generator(device="cpu")
        rng_cpu.manual_seed(self.seed_offset + idx)

        midi_params = torch.tensor(
            [int(torch.empty(1).random_(30, 127, generator=rng_cpu).item()) for _ in range(3)]
        )

        synth_params = torch.empty(self.num_out_features).uniform_(0, 1, generator=rng_cpu)

        audio_out = torch.empty(self.num_out_features).uniform_(0, 1, generator=rng_cpu)

        return midi_params, synth_params, audio_out, idx


class DummyModel(LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 4 * out_features), nn.ReLU(), nn.Linear(4 * out_features, out_features)
        )
        self.loss = nn.L1Loss()
        self.sampler_idx_last = -1
        self.training_from_checkpoint = False

    def forward(self, presets_emb):
        return self.model(presets_emb)

    def on_train_start(self) -> None:
        # for testing
        if self.training_from_checkpoint:
            assert self.sampler_idx_last == SAMPLER_LAST_IDX_TARGET

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        _, presets_emb, audio_emb, _ = batch
        out = self(presets_emb)
        loss = self.loss(out, audio_emb)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        _, _, _, idx = batch
        sampler_idx_last = (idx[-1] if idx.shape[0] > 1 else idx).item()
        if sampler_idx_last > 0:
            self.sampler_idx_last = sampler_idx_last

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.sampler_idx_last > 0:
            checkpoint["sampler_idx_last"] = self.sampler_idx_last

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.training_from_checkpoint = True
        if checkpoint.get("sampler_idx_last"):
            self.sampler_idx_last = checkpoint["sampler_idx_last"]


def test_custom_sequential_sampler(tmp_path):
    ###### Settings, dataset, and model
    tmp_dir = tmp_path / "checkpoints"
    tmp_dir.mkdir()

    seed_everything(1)

    dataset = DummyDataset(
        dataset_size=DATASET_SIZE, seed_offset=SEED_OFFSET, num_out_features=NUM_OUT_FEATURES
    )
    lit_model = DummyModel(in_features=NUM_OUT_FEATURES, out_features=NUM_OUT_FEATURES)

    ###### initial training
    sampler = SequentialSampler2(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        sampler=sampler,
        pin_memory=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=MAX_STEPS,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        deterministic=True,
        check_val_every_n_epoch=None,
    )
    trainer.fit(lit_model, dataloader)

    assert lit_model.sampler_idx_last == SAMPLER_LAST_IDX_TARGET

    trainer.save_checkpoint(tmp_dir / "dummy-ckpt.ckpt")

    ###### resume training
    start_idx = torch.load(tmp_dir / "dummy-ckpt.ckpt")["sampler_idx_last"] + 1
    sampler = SequentialSampler2(dataset, start_idx)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        sampler=sampler,
        pin_memory=False,
        drop_last=True,
    )
    assert dataloader.sampler.start_idx == SAMPLER_LAST_IDX_TARGET + 1

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=MAX_STEPS * 2,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        deterministic=True,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model=lit_model, train_dataloaders=dataloader, ckpt_path=str(tmp_dir / "dummy-ckpt.ckpt"))

    assert lit_model.sampler_idx_last == BATCH_SIZE * MAX_STEPS * 2 - 1
