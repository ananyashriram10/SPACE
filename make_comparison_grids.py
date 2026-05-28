#!/usr/bin/env python3
"""Generate per-artist comparison grids across all available methods."""

import os
import textwrap
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ARTISTS = [
    {
        "name": "Kelly McKernan",
        "csv": "data/kelly_prompts.csv",
        "baseline": "results/baseline/kelly",
        "methods": {
            "ESD-x": ("results/erased/kelly", "case"),
            "UCE": ("results/uce/uce-Kelly_McKernan", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Kelly_McKernan/samples", "order"),
            "SPACE": ("results/space/space-Kelly_McKernan", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Kelly_McKernan", "case"),
            "SPACE-v3": ("results/space_v3/space-Kelly_McKernan", "case"),
        },
    },
    {
        "name": "Van Gogh",
        "csv": "data/vangogh_prompts.csv",
        "baseline": "results/baseline/vangogh",
        "methods": {
            "ESD-x": ("results/erased/vangogh", "case"),
            "UCE": ("results/uce/uce-Van_Gogh", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples", "order"),
            "SPACE": ("results/space/space-Van_Gogh", "case"),
            "v2R e125": ("results/space_v2r_e125/space-Van_Gogh", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Van_Gogh", "case"),
            "v2R e225": ("results/space_v2r_e225/space-Van_Gogh", "case"),
            "v2R top35": ("results/space_v2r_e175_top35/space-Van_Gogh", "case"),
            "SPACE-v3": ("results/space_v3/space-Van_Gogh", "case"),
        },
    },
    {
        "name": "Tyler Edlin",
        "csv": "data/short_niche_art_prompts.csv",
        "artist_filter": "Tyler Edlin",
        "baseline": "results/baseline/tyler_edlin",
        "methods": {
            "ESD-x": ("results/erased/tyler_edlin", "case"),
            "UCE": ("results/uce/uce-Tyler_Edlin", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Tyler_Edlin/samples", "order"),
            "SPACE": ("results/space/space-Tyler_Edlin", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Tyler_Edlin", "case"),
            "SPACE-v3": ("results/space_v3/space-Tyler_Edlin", "case"),
        },
    },
    {
        "name": "Thomas Kinkade",
        "csv": "data/short_niche_art_prompts.csv",
        "artist_filter": "Thomas Kinkade",
        "baseline": "results/baseline/thomas_kinkade",
        "methods": {
            "ESD-x": ("results/erased/thomas_kinkade", "case"),
            "UCE": ("results/uce/uce-Thomas_Kinkade", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Thomas_Kinkade/samples", "order"),
            "SPACE": ("results/space/space-Thomas_Kinkade", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Thomas_Kinkade", "case"),
            "SPACE-v3": ("results/space_v3/space-Thomas_Kinkade", "case"),
        },
    },
    {
        "name": "Kilian Eng",
        "csv": "data/short_niche_art_prompts.csv",
        "artist_filter": "Kilian Eng",
        "baseline": "results/baseline/kilian_eng",
        "methods": {
            "ESD-x": ("results/erased/kilian_eng", "case"),
            "UCE": ("results/uce/uce-Kilian_Eng", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Kilian_Eng/samples", "order"),
            "SPACE": ("results/space/space-Kilian_Eng", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Kilian_Eng", "case"),
            "SPACE-v3": ("results/space_v3/space-Kilian_Eng", "case"),
        },
    },
    {
        "name": "Ajin: Demi Human",
        "csv": "data/short_niche_art_prompts.csv",
        "artist_filter": "Ajin: Demi Human",
        "baseline": "results/baseline/ajin_demi_human",
        "methods": {
            "ESD-x": ("results/erased/ajin", "case"),
            "UCE": ("results/uce/uce-Ajin_Demi_Human", "case"),
            "CA": ("results/concept_ablation_compvis/concept_ablation-Ajin_Demi_Human/samples", "order"),
            "SPACE": ("results/space/space-Ajin_Demi_Human", "case"),
            "v2R e175": ("results/space_v2r_e175/space-Ajin_Demi_Human", "case"),
            "SPACE-v3": ("results/space_v3/space-Ajin_Demi_Human", "case"),
        },
    },
    {
        "name": "Andy Warhol",
        "csv": "data/andy_warhol_prompts.csv",
        "baseline": "results/baseline/andy_warhol",
        "methods": {
            "ESD-x": ("results/erased/andy_warhol", "case"),
            "UCE":   ("results/uce/uce-Andy_Warhol", "case"),
            "CA":    ("results/concept_ablation_diffusers/concept_ablation-Andy_Warhol", "case"),
            "SPACE": ("results/space/space-Andy_Warhol", "case"),
        },
    },
]

OUT_DIR = "results/comparisons"
IMG_SIZE = 2.6
WRAP_WIDTH = 34


def load_prompts(cfg):
    df = pd.read_csv(cfg["csv"])
    if "artist_filter" in cfg:
        df = df[df["artist"].astype(str).str.strip() == cfg["artist_filter"]].reset_index(drop=True)
    return df[["case_number", "prompt"]].copy()


def path_for(folder, mode, case, order):
    if mode == "order":
        return os.path.join(folder, f"{order:05d}.png")
    return os.path.join(folder, f"{case}_0.png")


def load_img(path):
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def make_grid(cfg, method_names=None):
    df = load_prompts(cfg)
    method_items = [
        (name, folder, mode)
        for name, (folder, mode) in cfg["methods"].items()
        if os.path.isdir(folder) and (method_names is None or name in method_names)
    ]
    cols = [("Prompt", None, None), ("Vanilla", cfg["baseline"], "case")] + method_items
    rows = []
    for order, row in df.iterrows():
        case = int(row["case_number"])
        images = [load_img(path_for(folder, mode, case, order)) if folder else None for _name, folder, mode in cols[1:]]
        if any(img is not None for img in images):
            rows.append((order, case, str(row["prompt"]), images))

    if not rows:
        print(f"  [{cfg['name']}] No images found")
        return

    fig_w = 3.0 + IMG_SIZE * (len(cols) - 1)
    fig_h = 1.0 + IMG_SIZE * len(rows)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f0f0f")
    fig.suptitle(f"{cfg['name']} — Official Baseline Comparison", color="white", fontweight="bold", y=0.998)
    gs = gridspec.GridSpec(
        len(rows),
        len(cols),
        figure=fig,
        width_ratios=[1.25] + [1] * (len(cols) - 1),
        hspace=0.04,
        wspace=0.04,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.01,
    )

    for row_idx, (_order, case, prompt, images) in enumerate(rows):
        ax_p = fig.add_subplot(gs[row_idx, 0])
        ax_p.set_facecolor("#1a1a1a")
        ax_p.axis("off")
        ax_p.text(0.05, 0.52, "\n".join(textwrap.wrap(prompt, WRAP_WIDTH)), color="#e0e0e0", fontsize=7, va="center")
        ax_p.text(0.05, 0.97, f"#{case}", color="#777", fontsize=6, va="top")
        if row_idx == 0:
            ax_p.set_title("Prompt", color="#aaa", fontsize=9)

        for col_idx, img in enumerate(images, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.axis("off")
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "missing", color="#555", ha="center", va="center")
            if row_idx == 0:
                ax.set_title(cols[col_idx][0], color="#aaa", fontsize=9)

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = cfg["name"].lower().replace(" ", "_").replace(":", "").replace("-", "_")
    png_path = os.path.join(OUT_DIR, f"{slug}.png")
    pdf_path = os.path.join(OUT_DIR, f"{slug}.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{cfg['name']}] {len(rows)} rows, {len(cols)-1} image columns -> {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-artist comparison grids.")
    parser.add_argument("--only-artist", default=None, help="Generate only this artist name.")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method labels to include, e.g. ESD-x,UCE,CA.",
    )
    args = parser.parse_args()
    artist_configs = ARTISTS
    if args.only_artist:
        artist_configs = [cfg for cfg in ARTISTS if cfg["name"] == args.only_artist]
        if not artist_configs:
            raise SystemExit(f"Unknown artist: {args.only_artist}")
    method_names = None
    if args.methods:
        method_names = {name.strip() for name in args.methods.split(",") if name.strip()}

    print(f"\nGenerating comparison grids -> {OUT_DIR}/\n")
    for artist_cfg in artist_configs:
        make_grid(artist_cfg, method_names=method_names)
    print("\nDone.\n")
