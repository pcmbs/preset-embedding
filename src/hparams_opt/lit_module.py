"""
Lightning Module for the preset embedding framework.
"""
from typing import Any, Dict, Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn

from utils.misc import compute_mrr
from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PresetEmbeddingHp(LightningModule):
    """
    Lightning Module for hyperparameter optimization of the preset embedding framework.
    """

    def __init__(
        self,
        preset_encoder: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.preset_encoder = preset_encoder
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        self.mrr_preds = []
        self.mrr_targets = None

        # ModuleList() allows params of stateful modules to be move to the correct device
        # self.metrics = nn.ModuleList(metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.preset_encoder(x)

    def _model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preset, audio_embedding = batch
        preset_embedding = self(preset)
        return preset_embedding, audio_embedding

    def training_step(self, batch, batch_idx: int):
        preset_embedding, audio_embedding = self._model_step(batch)
        loss = self.loss(preset_embedding, audio_embedding)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.mrr_preds.clear()
        self.mrr_targets = None

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        preset, audio_embedding = batch
        if batch_idx == 0:
            self.mrr_targets = audio_embedding
        self.mrr_preds.append(self(preset))

    def on_validation_epoch_end(self) -> None:
        num_eval, preds_dim = self.mrr_targets.shape
        # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
        targets = self.mrr_targets.unsqueeze_(1)
        # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
        preds = torch.cat(self.mrr_preds, dim=1).view(num_eval, -1, preds_dim)
        mrr_score = compute_mrr(preds, targets, index=0, p=1)
        self.log("val/mrr", mrr_score, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(params=self.preset_encoder.parameters(), lr=self.lr)

        if self.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler(optimizer=optimizer)
        scheduler_config = self.scheduler_config
        scheduler_config["scheduler"] = scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
