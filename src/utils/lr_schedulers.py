"""
Script to build custom schedulers
"""

import torch
import torch.optim.lr_scheduler as S


def lin_cos_scheduler_builder(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    milestone: int,
    final_lr: float,
    linear_start_factor: float = 1,
    linear_end_factor: float = 1,
) -> S._LRScheduler:
    """
    Builds a cosine LR scheduler preceded by a linear LR scheduler,
    which can either act as a linear warmup or to keep the learning rate constant allowing for a longer
    exploration phase.

    Args
    - optimizer (torch.optim.Optimizer): the optimizer for which the LR scheduler is built
    - total_steps (int): the total number of training steps.
    - milestone (int): the number of steps after which the cosine LR scheduler is triggered.
    - final_lr (float): the final learning rate of the cosine LR scheduler.
    - linear_start_factor (float): the starting factor of the linear LR scheduler, to be multiplied with
    the initial LR. (Default: 1)
    - linear_end_factor (float): the ending factor of the linear LR scheduler, to be multiplied with
    the initial LR. (Default: 1)

    Returns
    - scheduler (torch.optim.lr_scheduler._LRScheduler): the LR scheduler with the linear LR scheduler as
    a torch sequential LR scheduler.
    """
    # Build the linear LR scheduler
    linear_scheduler = S.LinearLR(
        optimizer, start_factor=linear_start_factor, end_factor=linear_end_factor, total_iters=milestone
    )
    # Build the cosine LR scheduler
    cosine_scheduler = S.CosineAnnealingLR(optimizer, T_max=total_steps - milestone, eta_min=final_lr)
    # Combine both schedulers
    scheduler = S.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[milestone]
    )
    return scheduler
