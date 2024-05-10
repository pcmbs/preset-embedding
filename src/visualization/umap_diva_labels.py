from utils.visualization import umap_diva_labels

# Matplotlib stuff
FONT_SIZE = 30

# should be in ["mlp_oh", "hn_oh", "hn_pt", "hn_ptgru", "tfm"]
MODELS = ["tfm", "hn_ptgru", "mlp_oh"]

LABELS_TO_PLOT = (("Aggressive", "Soft"), ("Bright", "Dark"))

# for reproducibility (for UMAP random state and numpy random choice)
SEED = 42

# for generating preset embedding if required
DATASET_VERSION = 1
BATCH_SIZE = 512


if __name__ == "__main__":

    umap_diva_labels(
        models=MODELS,
        labels_to_plot=LABELS_TO_PLOT,
        font_size=FONT_SIZE,
        seed=SEED,
        dataset_version=DATASET_VERSION,
        batch_size=BATCH_SIZE,
    )
