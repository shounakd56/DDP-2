"""
visualize_transport_maps.py
---------------------------
Spatial visualization of the UOT transport plan over WSI patches.

For each selected slide we:
  1. encode all 5x patches with the trained CONCH-MUOT encoder,
  2. compute π = UOT(patch_features, concept_text_features),
  3. project π over the spatial grid recovered from patch filenames,
  4. render a publication-grade panel:
        ┌ slide thumbnail mosaic ┐
        │  + heatmap overlay     │  per top-K activated concept
        └────────────────────────┘

Why "stitched mosaic" instead of the raw WSI?
   We don't have access to the original .svs file inside this repo, but
   we DO have all the patch images on disk. Stitching them at small
   thumbnail size with their parsed (row, col) gives an excellent
   spatial reference for the heatmap. If filenames carry no parseable
   coordinates we fall back to a square grid.

Usage
-----
    python visualize_transport_maps.py \
        --checkpoint ./fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
        --main_module train_conch_rcc \
        --output_dir ./interpretability_outputs/transport_maps \
        --num_per_class 2 \
        --top_k_concepts 5 \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from PIL import Image

from extract_pi import (
    load_trained_model, get_concept_text_features,
    encode_slide_at_scale, compute_pi,
    load_split_with_paths, pick_slides_for_class, _make_eval_transform,
)
from coordinate_utils import (
    parse_patch_coordinates, build_mosaic, values_to_grid, normalise_coords,
)

warnings.filterwarnings("ignore")

# Publication-grade style
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


# ---------------------------------------------------------------------------
# Heatmap colormaps (used for diagnostic class-association coloring)
# ---------------------------------------------------------------------------
CLASS_CMAPS = {
    "KICH":   "Reds",
    "KIRP":   "Greens",
    "KIRC":   "Blues",
    "shared": "Purples",
}


def _grid_to_image(grid: np.ndarray, tile_size: int) -> np.ndarray:
    """Upsample a (n_rows, n_cols) heatmap to (n_rows*tile, n_cols*tile)
    by nearest-neighbour replication. Returns float32 array with NaN
    preserved for empty cells."""
    return np.kron(grid, np.ones((tile_size, tile_size), dtype=np.float32))


def _render_overlay(mosaic_img: Image.Image,
                    heat_grid_full: np.ndarray,
                    cmap_name: str = "magma",
                    alpha: float = 0.55) -> Image.Image:
    """Composite a heatmap over a mosaic. NaN cells become fully transparent."""
    base = np.asarray(mosaic_img.convert("RGBA")).astype(np.float32) / 255.0
    H, W, _ = base.shape

    # Pad/crop heat grid to match mosaic dims
    hh, hw = heat_grid_full.shape
    if (hh, hw) != (H, W):
        # Resize without smoothing (block effect is desirable here)
        from PIL import Image as _PIL
        heat_pil = _PIL.fromarray(np.nan_to_num(heat_grid_full, nan=0).astype(np.float32))
        heat_resized = heat_pil.resize((W, H), _PIL.Resampling.NEAREST)
        heat_full = np.array(heat_resized, dtype=np.float32)
        # rebuild NaN mask analogously
        nan_mask = np.isnan(heat_grid_full).astype(np.float32)
        nan_pil = _PIL.fromarray(nan_mask)
        nan_resized = np.array(nan_pil.resize((W, H), _PIL.Resampling.NEAREST))
        heat_full[nan_resized > 0.5] = np.nan
    else:
        heat_full = heat_grid_full

    valid = ~np.isnan(heat_full)
    if not valid.any():
        return mosaic_img.convert("RGBA")

    vmax = np.nanmax(heat_full)
    vmin = np.nanmin(heat_full[valid])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap(cmap_name)(norm(np.nan_to_num(heat_full, nan=vmin)))
    rgba = rgba.astype(np.float32)
    rgba[..., 3] = alpha * valid.astype(np.float32)

    # Alpha composite
    out = base * (1 - rgba[..., 3:4]) + rgba[..., :3] * rgba[..., 3:4]
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# One-slide visualization
# ---------------------------------------------------------------------------
def visualize_one_slide(model, slide, scale, ctx, out_dir,
                        top_k_concepts: int = 5,
                        max_patches: int = 1500,
                        tile_size: int = 48,
                        max_dim: int = 2400):
    """Generate transport-map figure for one slide. Saves PDF + PNG."""
    cfg = model.config
    sid = slide["slide_id"]
    cls = slide["class_name"]
    label = int(slide["label"])
    print(f"\n[slide] {sid}  (class={cls})")

    # 1) encode patches
    patch_feats, patch_paths = encode_slide_at_scale(
        model, slide, scale, max_patches=max_patches,
        transform=_make_eval_transform())

    if patch_feats.shape[0] == 0:
        print(f"  >> no patches loaded; skip"); return
    if patch_paths is None:
        print(f"  >> no raw images at scale '{scale}'; "
              f"skipping spatial visualization for this slide.")
        return

    print(f"  patches={patch_feats.shape[0]}, dim={patch_feats.shape[1]}")

    # 2) compute pi
    text_feats = ctx["_text_features_t"]
    pi_raw, pi_eff = compute_pi(model, patch_feats, text_feats,
                                apply_swr_gates=True, scale_idx=cfg.scales.index(scale))
    print(f"  pi shape = {pi_eff.shape}")

    # 3) coords + mosaic
    filenames = [Path(p).name for p in patch_paths]
    coords, mode = parse_patch_coordinates(filenames)
    print(f"  coordinate parsing: mode='{mode}'  "
          f"(rows×cols inferred from filenames)")
    mosaic, used_tile, n_rows, n_cols = build_mosaic(
        patch_paths, coords, tile_size=tile_size, max_dim=max_dim)
    coords_norm, _, _ = normalise_coords(coords)

    # 4) pick top-K concepts by total transport mass on this slide
    mass = pi_eff.sum(axis=0)        # (C,)
    top_idx = np.argsort(-mass)[:top_k_concepts]

    concept_names = ctx["concept_names"]
    class_assoc   = ctx["class_associations"]

    # ---------------- Figure layout ----------------
    cols = top_k_concepts + 1
    rows = 1
    fig_w = 2.6 * cols
    fig_h = 3.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if cols == 1:
        axes = [axes]

    axes[0].imshow(np.asarray(mosaic))
    axes[0].set_title(f"{sid}\nclass={cls}  ·  mode={mode}", fontsize=9)
    axes[0].axis("off")

    for k, ci in enumerate(top_idx):
        ax = axes[k + 1]
        cname = concept_names[ci]
        cassoc = class_assoc[ci]
        cmap = CLASS_CMAPS.get(cassoc, "magma")

        # Project π[:, ci] back onto the spatial grid
        grid = values_to_grid(pi_eff[:, ci], coords_norm, n_rows, n_cols, fill=np.nan)
        heat_full = _grid_to_image(grid, used_tile)
        overlay = _render_overlay(mosaic, heat_full, cmap_name=cmap, alpha=0.60)

        ax.imshow(np.asarray(overlay))
        ax.axis("off")
        # Title: rank, concept, association, total mass
        cname_short = cname if len(cname) <= 35 else cname[:33] + "…"
        ax.set_title(f"#{k+1}  [{cassoc}]\n{cname_short}\n"
                     f"mass={mass[ci]:.3g}", fontsize=8)

    fig.suptitle(
        f"Transport Map — {sid}   "
        f"(scale={scale}, top-{top_k_concepts} concepts shown)",
        fontsize=11, y=1.02)
    fig.tight_layout()

    base = Path(out_dir) / f"transport_{cls}_{sid}"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)
    print(f"  ✓ saved {base}.pdf / .png")

    # Also save the raw transport-mass distribution for this slide
    np.savez(base.with_suffix(".npz"),
             pi_raw=pi_raw, pi_eff=pi_eff,
             top_concept_indices=top_idx,
             concept_names=np.array(concept_names),
             class_associations=np.array(class_assoc),
             slide_id=sid, class_name=cls, label=label,
             coords=coords_norm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",       type=str, required=True)
    ap.add_argument("--main_module",      type=str, required=True)
    ap.add_argument("--output_dir",       type=str, required=True)
    ap.add_argument("--scale",            type=str, default="5x")
    ap.add_argument("--split",            type=str, default="test",
                    choices=["test", "val", "train"])
    ap.add_argument("--num_per_class",    type=int, default=2)
    ap.add_argument("--top_k_concepts",   type=int, default=5)
    ap.add_argument("--max_patches",      type=int, default=1500)
    ap.add_argument("--tile_size",        type=int, default=48)
    ap.add_argument("--max_dim",          type=int, default=2400)
    ap.add_argument("--gpu_id",           type=int, default=0)
    ap.add_argument("--seed",             type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg, train_mod = load_trained_model(args.checkpoint, args.main_module, device=device)

    # Concept text features (cached for this whole run)
    print("[ctx] computing concept text features ...")
    ctx = get_concept_text_features(model)
    print(f"[ctx] {len(ctx['concept_names'])} concepts; "
          f"learned_importance={'yes' if ctx['learned_importance'] is not None else 'no (no SWR layer found)'}")

    # Choose slides
    split_data = load_split_with_paths(cfg, train_mod, split=args.split)
    slides = pick_slides_for_class(split_data, num_classes=cfg.num_classes,
                                   num_per_class=args.num_per_class,
                                   rng_seed=args.seed)
    print(f"[pick] visualizing {len(slides)} slide(s) "
          f"({args.num_per_class}/class on '{args.split}')")

    if args.scale not in cfg.scales:
        raise ValueError(f"scale '{args.scale}' not in cfg.scales={cfg.scales}")

    for slide in slides:
        try:
            visualize_one_slide(model, slide, args.scale, ctx, args.output_dir,
                                top_k_concepts=args.top_k_concepts,
                                max_patches=args.max_patches,
                                tile_size=args.tile_size,
                                max_dim=args.max_dim)
        except Exception as e:
            print(f"  !! failed on {slide.get('slide_id', '?')}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n[done] all transport-map figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
