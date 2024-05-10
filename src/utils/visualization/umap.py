# pylint: disable=E1120:no-value-for-parameter
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from models.lit_module import PresetEmbeddingLitModule
from models.preset import model_zoo
from utils.synth import PresetHelper

load_dotenv()
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Matplotlib stuff
COLORS = plt.cm.Paired.colors
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["font.sans-serif"] = ["cmr10"]
plt.rcParams["mathtext.fontset"] = "stix"

MODEL_NAME_FORMAT = {
    "hn_pt": "HN-PT",
    "hn_ptgru": "HN-PTGRU",
    "hn_oh": "HN-OH",
    "mlp_oh": "MLP-OH",
    "tfm": "TFM",
}

DEXED_LABEL_COLOR_MAP = {
    "harmonic": COLORS[1],
    "percussive": COLORS[3],
    "sfx": COLORS[7],
}

EXPORT_DIR = PROJECT_ROOT / "reports" / "umap_projections"


def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except ValueError:
        return x


def get_labels(dataset_json: Dict, synth: str) -> List:
    assert synth in ["diva", "dexed"]
    labels_key = "labels" if synth == "dexed" else "character"
    labels = []
    for _, v in dataset_json.items():
        labels.extend(v["meta"].get(labels_key, []))
    labels = list(set(labels))
    labels.sort()
    return labels


def get_preset_ids_per_label(dataset_json: Dict, labels: List[str], synth: str) -> Dict[str, List]:
    assert synth in ["diva", "dexed"]
    labels_key = "labels" if synth == "dexed" else "character"
    preset_id = {k: [] for k in labels}
    for k, v in dataset_json.items():
        current_labels = v["meta"].get(labels_key, [])
        if synth == "dexed":
            # only get presets with a single label for dexed...
            if len(current_labels) == 1:
                preset_id[current_labels[0]].append(k)
        else:  # ...but this doesn't matter for diva
            for c in current_labels:
                preset_id[c].append(k)
    return preset_id


def get_mutexcl_diva_labels(dataset_json: Dict, labels: List[str]) -> Dict[str, List]:
    label_dict = {k: set(labels) for k in labels}
    for _, v in dataset_json.items():
        current_labels = v["meta"].get("character", [])
        if current_labels:
            for c, s in label_dict.items():
                if c in current_labels:
                    s.difference_update(current_labels)

    return {k: list(v) for k, v in label_dict.items()}


def get_diva_categories(
    dataset_json: Dict, included_categories: Sequence[str] = ("Bass", "Drums", "Keys", "Pads")
) -> Dict[str, List]:
    # How preset categories are gather:
    # Bass: Bass + Basses,
    # Drums: Drums + Drums & Percussive,
    # FX: FX,
    # Keys: Keys,
    # Leads: Leads,
    # Others: Other,
    # Pads: Pads,
    # Seq & Arp: Seq & Arp + Sequences & Arps
    # Stabs: Stabs + Plucks & Stabs
    categories = {k: [] for k in included_categories}
    # iterate over presets
    for k, v in dataset_json.items():
        # get current preset categories
        preset_categories = v["meta"].get("categories", [])

        for c in preset_categories:
            c = c.split(":")[0]
            # ignored categories: ["Other", "Seq & Arp", "Sequences & Arps"]
            if c in ["Bass", "Basses"]:
                if "Bass" in included_categories:
                    categories["Bass"].append(k)
            if c in ["Drums", "Drums & Percussive"]:
                if "Drums" in included_categories:
                    categories["Drums"].append(k)
            if c == "FX":
                if "FX" in included_categories:
                    categories["FX"].append(k)
            if c == "Keys":
                if "Keys" in included_categories:
                    categories["Keys"].append(k)
            if c == "Leads":
                if "Leads" in included_categories:
                    categories["Leads"].append(k)
            if c == "Others":
                if "Others" in included_categories:
                    categories["Others"].append(k)
            if c == "Pads":
                if "Pads" in included_categories:
                    categories["Pads"].append(k)
            if c in ["Seq & Arp", "Sequences & Arps"]:
                if "Seq & Arp" in included_categories:
                    categories["Sequences & Arps"].append(k)
            if c in ["Stabs", "Stabs & Plucks"]:
                if "Stabs" in included_categories:
                    categories["Stabs"].append(k)

    categories = {k: list(set(v)) for k, v in categories.items()}
    return categories


def umap_diva_labels(
    models: Sequence[str],
    labels_to_plot: Sequence[Tuple[str]] = (("Aggressive", "Soft"), ("Bright", "Dark")),
    font_size: int = 30,
    seed: int = 42,
    dataset_version: int = 1,
    batch_size: int = 512,
) -> None:

    EVAL_DIR = PROJECT_ROOT / "data" / "datasets" / "eval" / "diva_mn04_eval_v1"
    np.random.seed(seed)

    with open(EVAL_DIR / "presets.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f, object_hook=keystoint)

    with open(EVAL_DIR / "audio_embeddings.pkl", "rb") as f:
        audio_embeddings = torch.load(f)

    labels = get_labels(dataset_json, synth="diva")
    print("\nDiva labels:", labels)

    mutexcl_labels = get_mutexcl_diva_labels(dataset_json, labels)

    preset_ids = get_preset_ids_per_label(dataset_json, labels, synth="diva")

    print("\nNumber of presets per label:")
    for k, v in preset_ids.items():
        print(k, len(v))

    # mutually exclusive labels that are well discriminated by the audio model
    # labels_to_plot = [("Aggressive", "Soft"), ("Bright", "Dark")]
    for i, l in enumerate(labels_to_plot):
        if l[1] not in mutexcl_labels[l[0]]:
            print(f"\n{l} is not a mutual exclusive pair!")
            print("\nAvalaible mutually exclusive labels:")
            for k, v in mutexcl_labels.items():
                print(k, v)

    reducer = umap.UMAP(n_neighbors=75, min_dist=0.5, metric="euclidean", init="pca", random_state=seed)

    NUM_ROWS = len(labels_to_plot)
    NUM_COLS = len(models) + 1
    fig, ax = plt.subplots(
        nrows=NUM_ROWS, ncols=NUM_COLS, figsize=(4 * NUM_COLS, 4 * NUM_ROWS), layout="constrained"
    )
    # iterate over pairs of labels
    for i, (l1, l2) in enumerate(labels_to_plot):
        # dict to store the umap embeddings
        embeddings = {}

        # Extract embeddings and concatenate
        audio_emb_l1 = audio_embeddings[preset_ids[l1]].numpy()
        audio_emb_l2 = audio_embeddings[preset_ids[l2]].numpy()
        audio_emb = np.concatenate([audio_emb_l1, audio_emb_l2])
        # Compute the preset labels' start and end indices
        sep_idx = len(audio_emb_l1)
        # Fit and transform embeddings using UMAP
        u_audio = reducer.fit_transform(audio_emb)

        embeddings["ref"] = u_audio

        # iterate over the models to evaluate
        for j, m in enumerate(models):
            print(
                f"Computing UMAP embeddings for labels ({l1}, {l2}) using {MODEL_NAME_FORMAT[m]} model..."
            )
            if not (EVAL_DIR / "preset_embeddings" / f"{m}_embeddings.pkl").exists():
                filename = EVAL_DIR / "preset_embeddings" / f"{m}_embeddings.pkl"
                print(f"{filename} does not exists, generating and exporting embeddings...")
                generate_preset_embeddings(
                    synth="diva", model=m, batch_size=batch_size, dataset_version=dataset_version
                )

            with open(EVAL_DIR / "preset_embeddings" / f"{m}_embeddings.pkl", "rb") as f:
                preset_embeddings = torch.load(f)
            # Extract preset embeddings and concatenate
            preset_emb_l1 = preset_embeddings[preset_ids[l1]].numpy()
            preset_emb_l2 = preset_embeddings[preset_ids[l2]].numpy()
            preset_emb = np.concatenate([preset_emb_l1, preset_emb_l2])
            # Sanity check
            assert audio_emb.shape[0] == preset_emb.shape[0]
            assert len(audio_emb_l1) == len(preset_emb_l1)
            assert len(audio_emb_l2) == len(preset_emb_l2)

            # Fit and transform embeddings using UMAP
            u_preset = reducer.fit_transform(preset_emb)
            embeddings[m] = u_preset.copy()

        # Plot
        for a, (source, u_emb) in zip(ax[i], embeddings.items()):
            a.scatter(
                u_emb[:sep_idx, 0],
                u_emb[:sep_idx, 1],
                label=l1,
                s=30,
                alpha=0.5,
                color=COLORS[1],
                edgecolors="none",
            )
            a.scatter(
                u_emb[sep_idx:, 0],
                u_emb[sep_idx:, 1],
                label=l2,
                s=30,
                alpha=0.5,
                color=COLORS[7],
                edgecolors="none",
            )
            # only add legend and y labels on the reference (1st column)
            if source == "ref":
                a.set_ylabel(f"{l1} vs. {l2}", fontsize=font_size)

                legend_handles = [
                    plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[1]),
                    plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[7]),
                ]
                legend_labels = [l1, l2]
                ax_leg = plt.axes([0, 0.5 * (1 - i), 1, 0.5], facecolor=(1, 1, 1, 0))
                ax_leg.legend(
                    legend_handles,
                    legend_labels,
                    ncol=2,
                    frameon=False,
                    loc="lower left",
                    bbox_to_anchor=(0.02, -0.075 if i == 0 else -0.1),
                    fontsize="xx-large",
                )
                ax_leg.axis("off")

            # only add title on the first row
            if i == 0:
                if source == "ref":
                    a.set_title("Reference", fontsize=font_size)
                else:
                    a.set_title(f"{MODEL_NAME_FORMAT[source]}", fontsize=font_size)

            a.set_xticks([])
            a.set_yticks([])
            a.set_frame_on(False)

    # add line to separate the rows
    ax_outer = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    line = plt.Line2D([0.1, 0.9], [0.465, 0.465], color="black", linewidth=0.5)
    ax_outer.add_line(line)
    ax_outer.axis("off")

    # add a bit more space between the rows
    fig.get_layout_engine().set(hspace=0.18)

    plt.savefig(EXPORT_DIR / "umap_diva_labels.pdf", bbox_inches="tight")


def umap_dexed_labels(
    models: Sequence[str],
    harmonic_subset_size: int = 2000,
    font_size: int = 30,
    seed: int = 42,
    dataset_version: int = 1,
    batch_size: int = 512,
) -> None:
    EVAL_DIR = PROJECT_ROOT / "data" / "datasets" / "eval" / "dexed_mn04_eval_v1"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    with open(EVAL_DIR / "presets.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f, object_hook=keystoint)

    with open(EVAL_DIR / "audio_embeddings.pkl", "rb") as f:
        audio_embeddings = torch.load(f)

    labels = get_labels(dataset_json, synth="dexed")
    print("Dexed labels:", labels)

    preset_id = get_preset_ids_per_label(dataset_json, labels, synth="dexed")
    print("Number of presets per label:")
    for k, v in preset_id.items():
        print(k, len(v))

    # take a random subset of the harmonic presets
    rnd_harm_idx = np.random.choice(len(preset_id["harmonic"]), harmonic_subset_size, replace=False)
    preset_id["harmonic"] = [preset_id["harmonic"][i] for i in rnd_harm_idx]

    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=75, min_dist=0.99, metric="euclidean", init="pca", random_state=seed)

    # dict to store the umap embeddings
    embeddings = {}

    # Extract and concatenate audio embeddings
    audio_emb = np.concatenate([audio_embeddings[v].numpy() for v in preset_id.values()])
    # Compute the preset labels' start and end indices
    sep_idx = np.cumsum([0] + [len(v) for v in preset_id.values()])
    # Fit and transform embeddings using UMAP
    u_audio = reducer.fit_transform(audio_emb)
    embeddings["ref"] = u_audio

    # iterate over the models to evaluate
    for model in models:
        print(f"Computing UMAP embeddings for the {MODEL_NAME_FORMAT[model]} model...")
        if not (EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl").exists():
            filename = EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl"
            print(f"{filename} does not exists, generating and exporting embeddings...")
            generate_preset_embeddings(
                synth="diva", model=model, batch_size=batch_size, dataset_version=dataset_version
            )

        with open(EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl", "rb") as f:
            preset_embeddings = torch.load(f)

        # Extract preset embeddings and concatenate
        preset_emb = np.concatenate([preset_embeddings[v].numpy() for v in preset_id.values()])
        # Sanity checks
        assert audio_emb.shape[0] == preset_emb.shape[0]
        for v in preset_id.values():
            assert len(audio_embeddings[v]) == len(preset_embeddings[v])

        # Fit and transform embeddings using UMAP
        u_preset = reducer.fit_transform(preset_emb)
        embeddings[model] = u_preset

    # Plot
    fig, ax = plt.subplots(
        nrows=1, ncols=len(embeddings), figsize=(4 * len(embeddings), 4.2), layout="constrained"
    )
    for a, (source, u_emb) in zip(ax, embeddings.items()):
        for i, c in enumerate(preset_id.keys()):
            a.scatter(
                u_emb[sep_idx[i] : sep_idx[i + 1], 0],
                u_emb[sep_idx[i] : sep_idx[i + 1], 1],
                label=c,
                s=15,
                alpha=0.5,
                color=DEXED_LABEL_COLOR_MAP[c],
                edgecolors="none",
            )
        if source == "ref":
            legend_handles = [
                plt.Rectangle((0, 0), 2, 1, fill=True, color=v) for v in DEXED_LABEL_COLOR_MAP.values()
            ]
            legend_labels = list(DEXED_LABEL_COLOR_MAP)
            fig.legend(
                legend_handles,
                legend_labels,
                ncol=3,
                frameon=False,
                loc="outside lower left",
                fontsize="x-large",
            )
            a.set_title("Reference", fontsize=font_size)
        else:
            a.set_title(f"{MODEL_NAME_FORMAT[source]}", fontsize=font_size)
        a.set_xticks([])
        a.set_yticks([])
        a.axis("off")

    plt.savefig(EXPORT_DIR / "umap_dexed_labels.pdf", bbox_inches="tight")


def generate_preset_embeddings(
    synth: str, model: str, batch_size: int = 512, dataset_version: int = 1
) -> None:
    # Load eval configs to retrieve model configs and ckpt name
    eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "eval" / "model" / f"{synth}_{model}.yaml")["model"]
    model_cfg = {k: v for k, v in eval_cfg["cfg"].items() if k != "_target_"}
    ckpt_name = eval_cfg["ckpt_name"]

    # path to the eval dataset
    data_dir = PROJECT_ROOT / "data" / "datasets" / "eval" / f"{synth}_mn04_eval_v{dataset_version}"

    with open(data_dir / "configs.pkl", "rb") as f:
        configs = torch.load(f)

    with open(data_dir / "synth_parameters.pkl", "rb") as f:
        synth_parameters = torch.load(f)

    preset_helper = PresetHelper(
        synth_name=synth,
        parameters_to_exclude=configs["params_to_exclude"],
    )

    m_preset = getattr(model_zoo, model)(
        **model_cfg, out_features=configs["num_outputs"], preset_helper=preset_helper
    )
    lit_model = PresetEmbeddingLitModule.load_from_checkpoint(
        str(PROJECT_ROOT / "checkpoints" / ckpt_name), preset_encoder=m_preset
    )
    lit_model.to(DEVICE)
    lit_model.freeze()

    preset_embeddings = torch.empty(len(synth_parameters), configs["num_outputs"])

    dataset = TensorDataset(synth_parameters)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch[0].to(DEVICE)
        with torch.no_grad():
            preset_embeddings[i * batch_size : (i + 1) * batch_size] = lit_model(batch)

    (data_dir / "preset_embeddings").mkdir(parents=True, exist_ok=True)
    with open(data_dir / "preset_embeddings" / f"{model}_embeddings.pkl", "wb") as f:
        torch.save(preset_embeddings, f)
