# pylint: disable=W1203:logging-fstring-interpolation
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

log = logging.getLogger(__name__)


def _compute_ranks(preds: torch.Tensor, targets: torch.Tensor, index: int = 0, p: int = 1) -> float:
    assert preds.shape[0] == targets.shape[0]
    assert preds.shape[2] == targets.shape[2]
    assert targets.shape[1] == 1

    # For each evaluation: compute the distances between each prediction and the target,
    # then sort the distances in ascending order and get the rank of the prediction matching the target
    distances = torch.cdist(preds, targets, p=p).squeeze()  # shape: (num_eval, num_preds_per_eval)
    ranks = distances.argsort(dim=-1)
    ranks = torch.where(ranks == index)[1]
    return ranks, distances


def compute_mrr(preds: torch.Tensor, targets: torch.Tensor, index: int = 0, p: int = 1) -> float:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric using `torch.cdist`.

    Args
    - `preds` (torch.Tensor): predictions of shape (num_eval, num_preds_per_eval, preds_dim),
    where num_eval corresponds to the number reciprocal ranks to be computed, num_preds_per_eval corresponds
    to the number of predictions per per evaluation, and preds_dim is the dimension of the predictions
    - `targets` (torch.Tensor): target of shape (num_eval, 1, preds_dim). Note that targets.shape[1] should
    be 1 since there is a single target per evaluation.
    - `index` (int): index of the prediction to be considered as the target in each evaluation. (default: 0)
    - `p` (int): p value for the p-norm used to compute the distances between each prediction and the target
    in each evaluation. (default: 1)
    Returns
    - MRR score as float between 0 and 1

    """
    ranks, _ = _compute_ranks(preds, targets, index=index, p=p)
    return (1 / (ranks + 1)).mean().item()


def eval_mrr(model: nn.Module, dataset: Dataset, p: int = 1) -> float:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric using `torch.cdist` to be used for evaluation.
    The rank of each samples in the dataset is computed by considering all other samples in the dataset.

    Args
    - `model` (nn.Module): model to be evaluated
    - `dataset` (Dataset): dataset used for evaluation. Note that the first batch taken from this
    dataset will be used as targets in each evaluation
    - `p` (int): p value for the p-norm used to compute the distances between each prediction and the target
    in each evaluation. (default: 1)

    Returns
    - MRR score as float between 0 and 1
    - a dictionary {rank: [preset_ids]} where each key is the compute rank and each
    value is a list of preset ids that ended up with that rank
    """

    BATCH_SIZE = 512

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    # get model device
    device = next(model.parameters()).device

    preset_embeddings = torch.empty((len(dataset), dataset.embedding_dim), device=device)
    audio_embeddings = torch.empty((len(dataset), dataset.embedding_dim), device=device)

    # compute preset embeddings and retrieve audio embeddings
    model.eval()
    with torch.no_grad():
        for i, (preset, audio_emb) in enumerate(dataloader):
            audio_embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = audio_emb.to(device)
            preset_embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = model(preset.to(device))

    # compute the distance matrix between each pair of preset and audio embeddings
    distances = torch.cdist(preset_embeddings, audio_embeddings, p=p)
    # get the rank of the diagonal entries of the distance matrix which originate from the same preset
    rank_matrix = distances.argsort(dim=-1)
    ranks = torch.empty_like(rank_matrix.diag())
    for preset_id, row in enumerate(rank_matrix):
        ranks[preset_id] = torch.nonzero(row == preset_id).item()
    # compute the MRR
    mrr_score = (1 / (ranks + 1)).mean().item()
    # log results
    log.info(f"MRR score: {mrr_score}")

    # Create a dictionary in which the keys are ranks and the values are lists of
    # preset ids that ended up with that rank
    ranks_dict = {}
    for preset_id, rank in enumerate(ranks):
        rank = rank.item() + 1  # use 1-based ranks for analysis
        if rank not in ranks_dict:
            ranks_dict[rank] = [preset_id]
        else:
            ranks_dict[rank].append(preset_id)
    # sort dict key by ascending rank
    ranks_dict = dict(sorted(ranks_dict.items()))

    return mrr_score, ranks_dict
