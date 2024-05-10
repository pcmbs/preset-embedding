from utils.visualization import umap_dexed_labels


font_size = 21

# should be in ["mlp_oh", "hn_oh", "hn_pt", "hn_ptgru", "tfm"]
models = ["tfm", "hn_ptgru", "mlp_oh"]

# we take 2k harmonic presets out of 25k using numpy random choice
# since there are only 800 percussive and 1500 sfx presets
harmonic_subset_size = 2000

# for reproducibility (for UMAP random state and numpy random choice)
seed = 42

# for generating preset embedding if required
dataset_version = 1
batch_size = 512


if __name__ == "__main__":

    umap_dexed_labels(
        models=models,
        harmonic_subset_size=harmonic_subset_size,
        font_size=font_size,
        seed=seed,
        dataset_version=dataset_version,
        batch_size=batch_size,
    )
