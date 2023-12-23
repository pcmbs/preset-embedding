import torch


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
    assert preds.shape[0] == targets.shape[0]
    assert preds.shape[2] == targets.shape[2]
    assert targets.shape[1] == 1

    # For each evaluation: compute the distances between each prediction and the target,
    # then sort the distances in ascending order and get the rank of the prediction matching the target
    distances = torch.cdist(preds, targets, p=p).squeeze()  # shape: (num_eval, num_preds_per_eval)
    ranks = distances.argsort(dim=-1)
    ranks = torch.where(ranks == index)[1]
    return (1 / (ranks + 1)).mean().item()
