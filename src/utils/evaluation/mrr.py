# pylint: disable=W1203:logging-fstring-interpolation
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn

log = logging.getLogger(__name__)


def _compute_ranks(
    preds: torch.Tensor, targets: torch.Tensor, index: int = 0, p: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def hc_eval_mrr(model: nn.Module, dataset: Union[Dataset, Subset], p: int = 1) -> Tuple[float, List, Dict]:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric using `torch.cdist` to be used for evaluation.
    The rank of each samples in the dataset is computed by considering all other samples in the dataset.

    Args
    - `model` (nn.Module): model to be evaluated
    - `dataset` (Union[Dataset, Subset]): dataset used for evaluation.
    - `p` (int): p value for the p-norm used to compute the distances between each prediction and the target
    in each evaluation. (default: 1)

    Returns
    - MRR score as float between 0 and 1
    - a list containig the top-k MRR for k=1, 2, ..., 5
    - a dictionary {rank: [preset_ids]} where each key is a rank and each
    value is a list of preset ids that ended up with that rank
    """

    BATCH_SIZE = 512

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    # get model device
    device = next(model.parameters()).device
    if isinstance(dataset, Subset):
        embedding_dim = dataset.dataset.embedding_dim
    elif isinstance(dataset, Dataset):
        embedding_dim = dataset.embedding_dim
    else:
        raise ValueError("Unsupported dataset type")
    preset_embeddings = torch.empty((len(dataset), embedding_dim), device=device)
    audio_embeddings = torch.empty((len(dataset), embedding_dim), device=device)

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
    ranks_dict = _create_rank_dict(ranks)

    top_k_mrr = [top_k_mrr_score(ranks_dict, k=k) for k in range(1, 6)]
    for k, mrr in enumerate(top_k_mrr):
        log.info(f"Top-{k+1} MRR score: {mrr}")

    return mrr_score, top_k_mrr, ranks_dict


def _create_rank_dict(ranks: torch.Tensor) -> Dict:
    """
    Create a dictionary in which the keys are ranks and the values are lists of
    preset ids that ended up with that rank
    """
    ranks_dict = {}
    for i, rank in enumerate(ranks):
        rank = rank.item() + 1  # use 1-based ranks for analysis
        if rank not in ranks_dict:
            ranks_dict[rank] = [i]
        else:
            ranks_dict[rank].append(i)
    # sort dict key by ascending rank
    ranks_dict = dict(sorted(ranks_dict.items()))
    return ranks_dict


def rnd_presets_eval(
    model: nn.Module, dataset: Dataset, num_ranks: int = 256, p: int = 1
) -> Tuple[float, float, List, Dict]:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric over a randomly generated dataset
    to be used for evaluation. The number of ranking evaluation to compute the mean from (the
    resulting number of samples per evaluation is thus dataset_size // num_ranks) so
    num_ranks = 256 and dataset_size = 131072, corresponds to 256 ranking evaluations, each
    containing 1 target amongst 512 candidates

    Args
    - `model` (nn.Module): model to be evaluated
    - `dataset` (Dataset): dataset used for evaluation. Note that the first batch taken from this
    dataset will be used as targets in each evaluation
    - `num_evals` (int): number of ranking evaluations to compute the mean from. (default: 256)
    - `p` (int): p value for the p-norm used to compute the distances between each prediction and the target
    in each evaluation. (default: 1)

    Returns
    - L1 loss as float
    - MRR score as float between 0 and 1
    - a list containig the top-k MRR for k=1, 2, ..., 5
    - a dictionary {rank: [preset_ids]} where each key is a rank and each
    value is a list of preset ids that ended up with that rank
    """
    dataloader = DataLoader(dataset, batch_size=num_ranks, shuffle=False, num_workers=0, drop_last=True)

    # get model device
    device = next(model.parameters()).device

    mrr_preds = []
    mrr_targets = None
    losses = []

    # compute preset embeddings and retrieve audio embeddings
    model.eval()
    with torch.no_grad():
        for i, (preset, audio_emb) in enumerate(dataloader):
            audio_embeddings = audio_emb.to(device)
            preset_embeddings = model(preset.to(device))
            loss = nn.functional.l1_loss(preset_embeddings, audio_embeddings)
            losses.append(loss)
            if i == 0:
                mrr_targets = audio_embeddings
            mrr_preds.append(preset_embeddings)

    # Compute MRR, top-K MRR, and generate rank dictionary
    num_eval, preds_dim = mrr_targets.shape
    # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
    targets = mrr_targets.unsqueeze_(1)
    # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
    preds = torch.cat(mrr_preds, dim=1).view(num_eval, -1, preds_dim)
    ranks, _ = _compute_ranks(preds, targets, index=0, p=p)
    mrr_score = (1 / (ranks + 1)).mean().item()
    log.info(f"MRR score: {mrr_score}")

    ranks_dict = _create_rank_dict(ranks)

    top_k_mrr = [top_k_mrr_score(ranks_dict, k=k) for k in range(1, 6)]
    for k, mrr in enumerate(top_k_mrr):
        log.info(f"Top-{k+1} MRR score: {mrr}")

    # Compute L1 loss
    loss = sum(losses) / len(losses)
    log.info(f"L1 loss: {loss}")

    return loss, mrr_score, top_k_mrr, ranks_dict


def top_k_mrr_score(ranks: Dict, k: int = 5) -> float:
    num_targets = sum(len(val) for val in ranks.values())
    recriprocal_rank_at_k = sum(len(ranks.get(rank, [])) / rank for rank in range(1, k + 1))
    return recriprocal_rank_at_k / num_targets


def get_non_repeating_integers(N: int, K: int) -> List[float]:
    """
    Return a list of K non repeating integers between 0 and N.
    """
    assert K <= N
    arr = torch.arange(0, N)

    # Shuffle the array
    arr_shuffled = arr[torch.randperm(N)]

    # Take only the first K entries
    result = arr_shuffled[:K]

    return result.tolist()
