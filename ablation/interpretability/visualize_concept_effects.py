"""
visualize_concept_effects.py
----------------------------
Global / dataset-level analysis of what the concept prompts learned.

Produces five publication-quality figures:

  01_concept_importance.{pdf,png}
        Bar chart of learned concept_importance (sigmoid'd) and the
        diagnostic prior side-by-side. Color-coded by class association.

  02_concept_class_heatmap.{pdf,png}
        Mean transport mass per concept × per true class, computed by
        running the model in eval over the chosen split.
        Tells you which concepts each class is routed to on average.

  03_top_concepts_per_class.{pdf,png}
        For each class, a bar chart of the top-K concepts by mean
        transport mass + a check-mark if the concept's hard-coded
        class_association matches the class.

  04_prior_vs_learned_scatter.{pdf,png}
        Scatter of diagnostic_prior vs sigmoid(concept_importance),
        annotated by concept name. Shows whether learning followed,
        rejected, or refined the priors.

  05_concept_summary.csv
        Tabular summary of all concept-level statistics — useful for
        the supplementary material.

Usage
-----
    python visualize_concept_effects.py \
        --checkpoint ./fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
        --main_module train_conch_rcc \
        --output_dir ./interpretability_outputs/concept_effects \
        --scale 5x \
        --split test \
        --max_slides_per_class 30 \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from extract_pi import (
    load_trained_model, get_concept_text_features,
    encode_slide_at_scale, compute_pi,
    load_split_with_paths, _make_eval_transform,
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Class-association palette
ASSOC_COLOR = {
    "KICH":   "#E63946",
    "KIRP":   "#06A77D",
    "KIRC":   "#1D3557",
    "shared": "#9B9B9B",
}

CLASS_NAMES = ["KICH", "KIRP", "KIRC"]


def _save(fig, base):
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 01 — concept importance bar plot
# ---------------------------------------------------------------------------
def plot_concept_importance(ctx, out_dir):
    names = ctx["concept_names"]
    assoc = ctx["class_associations"]
    prior = ctx["diagnostic_prior"]
    learned = ctx["learned_importance"]

    n = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.32 * n + 1)))

    # --- Diagnostic prior ---
    ax = axes[0]
    order = np.argsort(-prior)
    bars = ax.barh(np.arange(n)[::-1],
                   prior[order],
                   color=[ASSOC_COLOR.get(assoc[i], "#888") for i in order],
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("Diagnostic prior weight")
    ax.set_title("(a) Diagnostic Prior  (concept_weights buffer)")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for b, v in zip(bars, prior[order]):
        ax.text(b.get_width() + 0.0005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=6.5)

    # --- Learned importance ---
    ax = axes[1]
    if learned is None:
        ax.text(0.5, 0.5,
                "No SWR layer in this model\n→ no learned concept_importance",
                ha="center", va="center", fontsize=11,
                transform=ax.transAxes,
                bbox=dict(facecolor="#FFE5A8", edgecolor="black", lw=0.5))
        ax.axis("off")
    else:
        order2 = np.argsort(-learned)
        bars = ax.barh(np.arange(n)[::-1],
                       learned[order2],
                       color=[ASSOC_COLOR.get(assoc[i], "#888") for i in order2],
                       edgecolor="black", linewidth=0.5)
        ax.set_yticks(np.arange(n)[::-1])
        ax.set_yticklabels([names[i] for i in order2], fontsize=7)
        ax.set_xlabel("σ(concept_importance)  — learned")
        ax.set_title("(b) Learned Concept Importance (after training)")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for b, v in zip(bars, learned[order2]):
            ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=6.5)

    legend = [Patch(facecolor=ASSOC_COLOR[k], edgecolor="black", label=k)
              for k in ["KICH", "KIRP", "KIRC", "shared"]]
    fig.legend(handles=legend, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)
    fig.suptitle("Concept Importance — Prior vs. Learned",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "01_concept_importance"))


# ---------------------------------------------------------------------------
# Plot 02 — concept × class heatmap
# ---------------------------------------------------------------------------
def plot_concept_class_heatmap(mass_per_class, ctx, out_dir):
    """mass_per_class : (num_classes, num_concepts) mean transport mass."""
    names = ctx["concept_names"]
    assoc = ctx["class_associations"]
    M = mass_per_class.copy()

    # Row-normalise so each class sums to 1 — emphasises distribution
    row_sums = M.sum(axis=1, keepdims=True).clip(min=1e-12)
    M_norm = M / row_sums

    # Sort concepts by total activation (descending)
    total_mass = M.sum(axis=0)
    order = np.argsort(-total_mass)
    M_norm_sorted = M_norm[:, order]

    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(names) + 2), 4.8))
    im = ax.imshow(M_norm_sorted, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xticks(range(len(names)))
    # color tick labels by association
    xtick_labels = []
    xtick_colors = []
    for i in order:
        nm = names[i]
        nm_short = nm if len(nm) <= 30 else nm[:28] + "…"
        xtick_labels.append(nm_short)
        xtick_colors.append(ASSOC_COLOR.get(assoc[i], "#888"))
    ax.set_xticklabels(xtick_labels, rotation=70, ha="right", fontsize=7)
    for label, color in zip(ax.get_xticklabels(), xtick_colors):
        label.set_color(color)
    ax.set_title("Mean (row-normalised) Transport Mass per Concept by True Class\n"
                 "(x-tick color = concept's class association)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("transport mass / row sum")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "02_concept_class_heatmap"))


# ---------------------------------------------------------------------------
# Plot 03 — top-K concepts per class
# ---------------------------------------------------------------------------
def plot_top_concepts_per_class(mass_per_class, ctx, out_dir, top_k=8):
    names = ctx["concept_names"]
    assoc = ctx["class_associations"]
    n_cls = mass_per_class.shape[0]

    fig, axes = plt.subplots(1, n_cls, figsize=(5.0 * n_cls, 5.0))
    if n_cls == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        order = np.argsort(-mass_per_class[c])[:top_k]
        vals = mass_per_class[c][order]
        labels_short = []
        colors = []
        for i in order:
            nm = names[i]
            tag = "✓" if assoc[i] == CLASS_NAMES[c] else (
                  " " if assoc[i] == "shared" else "✗")
            labels_short.append(f"{tag} {nm}" if len(nm) <= 35 else f"{tag} {nm[:33]}…")
            colors.append(ASSOC_COLOR.get(assoc[i], "#888"))
        bars = ax.barh(range(top_k)[::-1], vals,
                       color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(top_k)[::-1])
        ax.set_yticklabels(labels_short, fontsize=8)
        ax.set_xlabel("mean transport mass")
        ax.set_title(f"Top {top_k} concepts for class {CLASS_NAMES[c]}\n"
                     "(✓ matches class assoc.,  ✗ different,  blank = shared)",
                     fontsize=9)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:.3g}", va="center", fontsize=7)
    fig.suptitle("Most Activated Concepts per True Class", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "03_top_concepts_per_class"))


# ---------------------------------------------------------------------------
# Plot 04 — prior vs learned scatter
# ---------------------------------------------------------------------------
def plot_prior_vs_learned(ctx, out_dir):
    if ctx["learned_importance"] is None:
        return
    names = ctx["concept_names"]
    assoc = ctx["class_associations"]
    prior = ctx["diagnostic_prior"]
    learned = ctx["learned_importance"]

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    for i, nm in enumerate(names):
        ax.scatter(prior[i], learned[i],
                   color=ASSOC_COLOR.get(assoc[i], "#888"),
                   s=70, edgecolor="black", linewidth=0.6)
        nm_short = nm if len(nm) <= 28 else nm[:26] + "…"
        ax.annotate(nm_short, (prior[i], learned[i]),
                    fontsize=6.5, alpha=0.85,
                    xytext=(4, 3), textcoords="offset points")

    # Reference: prior min/max line
    pmin, pmax = float(prior.min()), float(prior.max())
    ax.set_xlabel("Diagnostic prior weight (concept_weights buffer)")
    ax.set_ylabel("σ(learned concept_importance)")
    ax.set_title("Prior vs. Learned Concept Importance\n"
                 "Above-diagonal ⇒ training upweighted the concept; "
                 "below ⇒ downweighted")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    legend = [Patch(facecolor=ASSOC_COLOR[k], edgecolor="black", label=k)
              for k in ["KICH", "KIRP", "KIRC", "shared"]]
    ax.legend(handles=legend, loc="best", title="class association")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "04_prior_vs_learned_scatter"))


# ---------------------------------------------------------------------------
# Aggregation pass: compute mass_per_class
# ---------------------------------------------------------------------------
@torch.no_grad()
def aggregate_mass_per_class(model, ctx, split_data, scale, num_classes,
                             max_per_class=30, max_patches=1200):
    transform = _make_eval_transform()
    by_class = defaultdict(list)
    for it in split_data:
        by_class[int(it["label"])].append(it)

    n_concepts = len(ctx["concept_names"])
    sum_mass = np.zeros((num_classes, n_concepts), dtype=np.float64)
    n_slides = np.zeros(num_classes, dtype=np.int64)

    text_feats = ctx["_text_features_t"]
    scale_idx = model.config.scales.index(scale)

    for c in range(num_classes):
        slides = by_class.get(c, [])[:max_per_class]
        print(f"\n[agg] class={CLASS_NAMES[c]} : aggregating over {len(slides)} slide(s)")
        for it in slides:
            try:
                pf, _ = encode_slide_at_scale(
                    model, it, scale, max_patches=max_patches, transform=transform)
                if pf.shape[0] == 0:
                    continue
                _, pi_eff = compute_pi(model, pf, text_feats,
                                       apply_swr_gates=True, scale_idx=scale_idx)
                # Mean over patches → distribution of concept activation for this slide
                slide_mass = pi_eff.mean(axis=0)
                sum_mass[c] += slide_mass
                n_slides[c] += 1
            except Exception as e:
                print(f"  !! {it['slide_id']}: {e}")
                continue
    mean_mass = sum_mass / n_slides[:, None].clip(min=1)
    return mean_mass.astype(np.float32), n_slides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",          type=str, required=True)
    ap.add_argument("--main_module",         type=str, required=True)
    ap.add_argument("--output_dir",          type=str, required=True)
    ap.add_argument("--scale",               type=str, default="5x")
    ap.add_argument("--split",               type=str, default="test",
                    choices=["test", "val", "train"])
    ap.add_argument("--max_slides_per_class", type=int, default=30,
                    help="Cap on slides used to estimate mean transport mass.")
    ap.add_argument("--max_patches",         type=int, default=1200)
    ap.add_argument("--top_k",               type=int, default=8,
                    help="Top-K concepts shown in per-class bar plots.")
    ap.add_argument("--gpu_id",              type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg, train_mod = load_trained_model(args.checkpoint, args.main_module, device=device)

    ctx = get_concept_text_features(model)
    print(f"[ctx] {len(ctx['concept_names'])} concepts; "
          f"learned_importance={'yes' if ctx['learned_importance'] is not None else 'no'}")

    if args.scale not in cfg.scales:
        raise ValueError(f"scale '{args.scale}' not in cfg.scales={cfg.scales}")

    # Plot 01 — global importance — needs no inference
    print("\n[plot 01] concept importance (prior vs learned) ...")
    plot_concept_importance(ctx, args.output_dir)

    # Aggregation pass
    split_data = load_split_with_paths(cfg, train_mod, split=args.split)
    if not split_data:
        print(f"[abort] no slides in split '{args.split}'"); return
    mass_per_class, n_slides_per_class = aggregate_mass_per_class(
        model, ctx, split_data, args.scale,
        num_classes=cfg.num_classes,
        max_per_class=args.max_slides_per_class,
        max_patches=args.max_patches)
    print(f"\n[agg] slides used per class: "
          f"{dict(zip(CLASS_NAMES, n_slides_per_class.tolist()))}")

    # Plot 02-03 — class-level concept analysis
    print("[plot 02] concept × class heatmap ...")
    plot_concept_class_heatmap(mass_per_class, ctx, args.output_dir)
    print("[plot 03] top-K concepts per class ...")
    plot_top_concepts_per_class(mass_per_class, ctx, args.output_dir, top_k=args.top_k)

    # Plot 04 — prior vs learned scatter
    print("[plot 04] prior vs learned scatter ...")
    plot_prior_vs_learned(ctx, args.output_dir)

    # CSV summary
    print("[csv]   concept_summary.csv ...")
    rows = []
    for i, nm in enumerate(ctx["concept_names"]):
        rec = {
            "concept":            nm,
            "class_association":  ctx["class_associations"][i],
            "diagnostic_prior":   float(ctx["diagnostic_prior"][i]),
        }
        if ctx["learned_importance"] is not None:
            rec["learned_importance"] = float(ctx["learned_importance"][i])
        for c, cn in enumerate(CLASS_NAMES):
            rec[f"mean_mass_{cn}"] = float(mass_per_class[c, i])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "05_concept_summary.csv"), index=False)

    # Save raw mass tensor too (for later re-plotting)
    np.savez(os.path.join(args.output_dir, "concept_effect_data.npz"),
             mass_per_class=mass_per_class,
             concept_names=np.array(ctx["concept_names"]),
             class_associations=np.array(ctx["class_associations"]),
             diagnostic_prior=ctx["diagnostic_prior"],
             learned_importance=ctx["learned_importance"]
                                if ctx["learned_importance"] is not None
                                else np.array([np.nan]),
             n_slides_per_class=n_slides_per_class)

    print(f"\n[done] all concept-effect figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
