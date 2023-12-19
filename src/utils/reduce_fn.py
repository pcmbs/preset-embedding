"""
Reduction functions.
"""

import torch


def flatten(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Flatten the input tensor along along the batch axis.

    Args
    - `embeddings` (torch.Tensor): The input tensor to be flattened.

    Returns
    - `torch.Tensor`: Flattened tensor).
    """
    return embeddings.flatten(start_dim=1)


def avg_channel_pooling(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling over the channel dimension of the input tensor.

    Args
    - `embeddings` (torch.Tensor): The input tensor.

    Returns
    - `torch.Tensor`: The output tensor.
    """
    return embeddings.mean(-2)


def avg_time_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the average pooling of the given embeddings.
    Note that when applied to a ViT-based model,
    this operation is equivalent to the average pooling over the patches.

    Args
    - `embeddings` (torch.Tensor): The input embeddings tensor.

    Returns:
        `torch.Tensor: Output tensor.
    """
    return embeddings.mean(-1)


def max_channel_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling over the channel dimension of the input tensor.

    Args
    - `embeddings` (torch.Tensor): The input tensor.

    Returns
    - `torch.Tensor`: The output tensor.
    """
    return embeddings.amax(-2)


def max_time_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the max pooling of the given embeddings.
    Note that when applied to a ViT-based model,
    this operation is equivalent to the max pooling over the patches.

    Args
    - `embeddings` (torch.Tensor): The input embeddings tensor.

    Returns:
        `torch.Tensor: Output tensor.
    """
    return embeddings.amax(-1)


if __name__ == "__main__":
    embeddings = torch.rand((10, 128, 500))

    print("original:", embeddings.shape)
    print("flatten:", flatten(embeddings).shape)
    print("global_avg_pool_channel:", avg_channel_pooling(embeddings).shape)
    print("global_avg_pool_time:", avg_time_pool(embeddings).shape)
    print("global_max_pool_channel:", max_channel_pool(embeddings).shape)
    print("global_max_pool_time:", max_time_pool(embeddings).shape)
