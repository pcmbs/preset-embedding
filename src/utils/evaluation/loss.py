import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

log = logging.getLogger(__name__)


def compute_l1(model: nn.Module, dataset: Dataset) -> float:
    """
    Function computing the L1 loss using `torch.cdist` to be used for evaluation.

    Args
    - `model` (nn.Module): model to be evaluated
    - `dataset` (Dataset): dataset used for evaluation.

    Returns
    - L1 loss as float
    """
    BATCH_SIZE = 512

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    # get model device
    device = next(model.parameters()).device

    # compute preset embeddings and retrieve audio embeddings
    preset_embeddings = torch.empty((len(dataset), dataset.embedding_dim), device=device)
    audio_embeddings = torch.empty((len(dataset), dataset.embedding_dim), device=device)

    # compute preset embeddings and retrieve audio embeddings
    model.eval()
    with torch.no_grad():
        for i, (preset, audio_emb) in enumerate(dataloader):
            audio_embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = audio_emb.to(device)
            preset_embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = model(preset.to(device))

    loss = nn.functional.l1_loss(preset_embeddings, audio_embeddings)
    log.info(f"L1 loss: {loss}")  # pylint: disable=W1203
    return loss
