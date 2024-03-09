from typing import List
import torch
from torch import nn

from utils.synth import PresetHelper

# TODO: move to test


class NonCatParamsEmb1(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Linear(1, L * C, bias=False)
        self.L = L
        self.C = C

    def init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        x = self.emb(x)  # shape (B, N, 1) -> (B, N, N * d)
        # indices of shape (N, d) where indices[i, j] = i * d + j
        # this is equivalent to torch.stack([x[:, i, i*d : (i+1)*d]] for i in range(N)], dim=1)
        indices = torch.arange(self.C) + torch.arange(self.L).view(-1, 1) * self.C
        x = x[:, torch.arange(self.L).view(-1, 1), indices]
        return x


class NonCatParamsEmbOpt(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(L, C))
        self.L = L
        self.C = C

    def init_weights(self, w: nn.ModuleList):
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, x):
        # emb * x: (1, N, C) * (B, N, 1) -> (B, N, C)
        # emb i-th row contains the linear projection of the i-th noncat parameter
        # and thus get multiplied by the i-th noncat parameters
        return self.emb * x.view(-1, self.L, 1)


class NonCatParamsEmbNaive(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.ModuleList([nn.Linear(1, C, bias=False) for _ in range(L)])
        self.L = L
        self.C = C

    def forward(self, x):
        return torch.stack([emb(x[:, i]) for i, emb in enumerate(self.emb)], dim=1)


class CatParamsEmb(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Embedding


if __name__ == "__main__":
    from timeit import default_timer as timer

    from torch.utils.data import DataLoader, TensorDataset

    BATCH_SIZE = 512
    NUM_NONCAT_P = 75
    EMB_DIM = 128

    dataset = TensorDataset(torch.randn((int(BATCH_SIZE * 1e2), NUM_NONCAT_P, 1)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedder_naive = NonCatParamsEmbNaive(NUM_NONCAT_P, EMB_DIM)
    embedder_1 = NonCatParamsEmb1(NUM_NONCAT_P, EMB_DIM)
    embedder_opt = NonCatParamsEmbOpt(NUM_NONCAT_P, EMB_DIM)

    with torch.no_grad():  # Ensure no gradients are tracked
        for i, linear_layer in enumerate(embedder_naive.emb):
            embedder_1.emb.weight[i * EMB_DIM : (i + 1) * EMB_DIM, :] = linear_layer.weight
            embedder_opt.emb[i, :] = linear_layer.weight.squeeze()

    start = timer()
    for batch in loader:
        out_naive = embedder_naive(batch[0])
    end = timer()
    print(f"EmbedNaive: {end - start:.5f}s")

    start = timer()
    for batch in loader:
        out_1 = embedder_1(batch[0])
    end = timer()
    print(f"Embed1: {end - start:.5f}s")

    start = timer()
    for batch in loader:
        out_opt = embedder_opt(batch[0])
    end = timer()
    print(f"EmbedOpt: {end - start:.5f}s")

    print(f"Embed1 vs. EmbedOpt: {torch.allclose(out_1, out_opt)}")
    print(f"Embed1 vs. EmbedNaive: {torch.allclose(out_1, out_naive)}")
    print(f"Embed2 vs. EmbedNaive: {torch.allclose(out_opt, out_naive)}")

    # embedder = NonCatParamsEmbedding(NUM_NONCAT_P, EMB_DIM)
    # x = torch.randn(BATCH_SIZE, NUM_NONCAT_P, 1)  # Example input with batch size 32
    # output = embedder(x)
    # print(output.shape)  # Should print: torch.Size([32, 150, 128])
