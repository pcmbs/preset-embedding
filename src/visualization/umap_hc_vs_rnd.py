import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import torch
import umap

load_dotenv()
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["font.sans-serif"] = ["cmr10"]
plt.rcParams["mathtext.fontset"] = "stix"
COLORS = plt.cm.Paired.colors
FONT_SIZE = 21


EVAL_DIR = PROJECT_ROOT / "data" / "datasets" / "eval"
EXPORT_DIR = PROJECT_ROOT / "reports" / "umap_projections"

SEED = 42

SUBSET_SIZE = {"dexed": 30_000, "diva": 10_000}


SYNTH = ["dexed", "diva"]
TITLES = {"dexed": "Dexed", "diva": "Diva"}

if __name__ == "__main__":
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    real_data_dict = {}
    synthetic_data_dict = {}
    for synth in SYNTH:
        with open(EVAL_DIR / f"{synth}_mn04_eval_v1" / "audio_embeddings.pkl", "rb") as f:
            real_data_dict[synth] = torch.load(f).numpy()

        with open(
            EVAL_DIR / f"{synth}_mn04_size=131072_seed=600_test_v1" / "audio_embeddings.pkl", "rb"
        ) as f:
            synthetic_data_dict[synth] = torch.load(f).numpy()
            num_samples = SUBSET_SIZE[synth]
            rnd_indexes = np.random.choice(len(synthetic_data_dict[synth]), num_samples, replace=False)
            synthetic_data_dict[synth] = synthetic_data_dict[synth][rnd_indexes]

    reducer = umap.UMAP(n_neighbors=75, min_dist=0.99, metric="euclidean", init="pca", random_state=SEED)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4.2), layout="constrained")

    for a, synth in zip(ax, SYNTH):
        print(f"Computing UMAP projection for {synth}...")

        real_data = real_data_dict[synth]
        synthetic_data = synthetic_data_dict[synth]

        # Concatenate audio embeddings
        embeddings = np.concatenate([real_data, synthetic_data])
        sep_idx = len(real_data)

        # Fit and transform embeddings using UMAP
        u_embeddings = reducer.fit_transform(embeddings)
        # u_embeddings = np.random.rand(embeddings.shape[0], 2) # dummy for plot debugging

        a.scatter(
            u_embeddings[:sep_idx, 0],
            u_embeddings[:sep_idx, 1],
            label="hand-crafted",
            s=3,
            alpha=0.5,
            color=COLORS[1],
            edgecolors="none",
        )
        a.scatter(
            u_embeddings[sep_idx:, 0],
            u_embeddings[sep_idx:, 1],
            label="synthetic",
            s=3,
            alpha=0.5,
            color=COLORS[7],
            edgecolors="none",
        )
        if synth == "dexed":
            legend_handles = [
                plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[1]),
                plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[7]),
            ]
            legend_labels = ["hand-crafted", "synthetic"]
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="outside lower left",
                fontsize="x-large",
                ncol=2,
            )
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(TITLES[synth], fontsize=FONT_SIZE)
        a.set_frame_on(False)

    plt.savefig(EXPORT_DIR / "umap_hc_vs_syn_presets.pdf", bbox_inches="tight")
