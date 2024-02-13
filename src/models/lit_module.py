"""
Lightning Module for the preset embedding framework.
"""
from typing import Any, Dict, Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn

from utils.evaluation import compute_mrr
from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PresetEmbeddingLitModule(LightningModule):
    """
    Lightning Module for the preset embedding framework.
    """

    def __init__(
        self,
        audio_feature_extractor: nn.Module,
        preset_encoder: nn.Module,
        loss: nn.Module,
        # metrics: Sequence[nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        wandb_watch_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.audio_fe = audio_feature_extractor
        self.preset_encoder = preset_encoder
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        self.wandb_watch_args = wandb_watch_args

        self.sampler_idx_last = -1

        self.mrr_preds = []
        self.mrr_targets = None

        # ModuleList() allows params of stateful modules to be move to the correct device
        # self.metrics = nn.ModuleList(metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.preset_encoder(x)

    def _model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preset, audio, _ = batch
        self.audio_fe.eval()
        with torch.no_grad():
            audio_embedding = self.audio_fe(audio).to(self.device)
        preset_embedding = self.preset_encoder(preset)
        return preset_embedding, audio_embedding

    def on_train_start(self) -> None:
        if not isinstance(self.logger, WandbLogger) or self.wandb_watch_args is None:
            log.info("Skipping watching model.")
        else:
            self.logger.watch(
                self.preset_encoder,
                log=self.wandb_watch_args["log"],
                log_freq=self.wandb_watch_args["log_freq"],
                log_graph=False,
            )

    def training_step(self, batch, batch_idx: int):
        preset_embedding, audio_embedding = self._model_step(batch)
        loss = self.loss(preset_embedding, audio_embedding)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        _, _, idx = batch
        sampler_idx_last = (idx[-1] if idx.shape[0] > 1 else idx).item()
        if sampler_idx_last > 0:
            self.sampler_idx_last = sampler_idx_last

    def on_train_end(self) -> None:
        if isinstance(self.logger, WandbLogger) and self.wandb_watch_args is not None:
            self.logger.experiment.unwatch(self.preset_encoder)

    def on_validation_epoch_start(self) -> None:
        self.mrr_preds.clear()
        self.mrr_targets = None

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        preset, audio_embedding = batch
        if batch_idx == 0:
            self.mrr_targets = audio_embedding
        self.mrr_preds.append(self.preset_encoder(preset))

    def on_validation_epoch_end(self) -> None:
        num_eval, preds_dim = self.mrr_targets.shape
        # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
        targets = self.mrr_targets.unsqueeze_(1)
        # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
        preds = torch.cat(self.mrr_preds, dim=1).view(num_eval, -1, preds_dim)
        mrr_score = compute_mrr(preds, targets, index=0, p=1)
        self.log("val/mrr", mrr_score, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(params=self.preset_encoder.parameters())

        if self.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler(optimizer=optimizer)
        scheduler_config = self.scheduler_config
        scheduler_config["scheduler"] = scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove the pre-trained audio model weights from the state_dict
        # to save time and memory since they can be downloaded at runtime
        state_dict_keys = list(checkpoint["state_dict"].keys())
        for key in state_dict_keys:
            if key.startswith("audio_fe"):
                del checkpoint["state_dict"][key]
        if self.sampler_idx_last > 0:
            checkpoint["sampler_idx_last"] = self.sampler_idx_last

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # include the pre-trained audio model weights in the loaded checkpoint's state_dict
        checkpoint["state_dict"].update({f"audio_fe.{k}": v for k, v in self.audio_fe.state_dict().items()})
        if checkpoint.get("sampler_idx_last"):
            self.sampler_idx_last = checkpoint["sampler_idx_last"]
