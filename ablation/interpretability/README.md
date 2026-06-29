# Interpretability Suite

Two scripts that produce paper-grade visualizations of what a trained
MUOT-CONCH model has learned about pathology concepts on TCGA-RCC:

1. **`visualize_transport_maps.py`** — for selected slides, overlays
   the UOT transport plan π[patch, concept] on a stitched mosaic of
   the slide's 5× patches. Answers the question *"which tissue area
   drove the routing to which concept?"*.

2. **`visualize_concept_effects.py`** — global / dataset-level analysis
   of concept importance: which concepts the model upweighted, what each
   class routes to on average, and whether the learned importance
   followed or contradicted the diagnostic priors.

## Files in this folder

| File | Purpose |
|---|---|
| `extract_pi.py` | Shared model-loading + π-extraction helpers |
| `coordinate_utils.py` | Parses `(row, col)` from patch filenames; builds mosaics |
| `visualize_transport_maps.py` | Per-slide spatial heatmaps |
| `visualize_concept_effects.py` | Global concept analysis plots |

## Quickstart

Run from the directory that contains `train_conch_rcc.py`,
`ablation_configs.py`, etc. The interpretability folder must be on the
import path; the simplest way is to run the scripts from inside it:

```bash
cd interpretability/
```

### Step 1 — Per-slide transport maps

```bash
python visualize_transport_maps.py \
    --checkpoint   ../fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
    --main_module  train_conch_rcc \
    --output_dir   ../interpretability_outputs/transport_maps \
    --num_per_class 2 \
    --top_k_concepts 5 \
    --gpu_id 0
```

For each picked slide you get:

- `transport_<class>_<slide_id>.pdf/png` — a single-row figure
  containing the **mosaic** plus the **top-K activated concepts** rendered
  as colored heatmaps over the same mosaic. Heatmap color encodes the
  concept's class association: red=KICH, green=KIRP, blue=KIRC,
  purple=shared.
- `transport_<class>_<slide_id>.npz` — the raw `pi_raw`, `pi_eff`,
  `coords`, etc. for re-plotting in case you want to customize layouts
  later.

### Step 2 — Global concept-effect plots

```bash
python visualize_concept_effects.py \
    --checkpoint   ../fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
    --main_module  train_conch_rcc \
    --output_dir   ../interpretability_outputs/concept_effects \
    --scale 5x \
    --split test \
    --max_slides_per_class 30 \
    --gpu_id 0
```

Produces:

| Figure | Shows |
|---|---|
| `01_concept_importance.{pdf,png}` | Bar chart of diagnostic prior vs learned `concept_importance`, color-coded by class association |
| `02_concept_class_heatmap.{pdf,png}` | Mean (row-normalised) transport mass per concept × per true class (concept tick labels are colored by their hard-coded class association) |
| `03_top_concepts_per_class.{pdf,png}` | For each class, top-K concepts by mean activation, with a ✓ / ✗ symbol indicating whether each concept's class-association matches |
| `04_prior_vs_learned_scatter.{pdf,png}` | Scatter of prior vs learned importance — above-diagonal concepts were upweighted by training, below-diagonal were downweighted |
| `05_concept_summary.csv` | Tabular summary of every concept's prior, learned weight, and per-class mean mass |
| `concept_effect_data.npz` | Raw arrays for re-plotting |

## How patch filenames are parsed

The transport-map script needs `(row, col)` for each patch to build a
mosaic. It tries the following patterns on each filename **stem**, in
order:

| Pattern | Example | Result |
|---|---|---|
| `r<row>_c<col>` | `r17_c42.jpeg` | `(17, 42)` |
| `x<col>_y<row>` | `x42_y17.png` | `(17, 42)` *(x↔col, y↔row)* |
| Last two integers | `tile_17_42.jpeg`, `slide_TCGA-XX-XXXX_17_42.png` | `(17, 42)` |
| (none parsed) | `tile_a.jpeg`, `tile_b.jpeg`, ... | `square grid fallback` |

The detection mode (`'parsed'` vs `'grid'`) is printed to stdout per
slide and shown in the figure subtitle. If almost all your patches end
up on a square grid you may want to write a tiny custom parser; see
`coordinate_utils.py`.

## What it costs

- **Transport maps**: one full ViT pass on the slide's 5× patches.
  About 30s per slide on an A100; ~3 minutes for 6 slides
  (2 per class).
- **Concept effects**: one ViT pass per slide × `max_slides_per_class`.
  With 30 slides per class × 3 classes = 90 ViT passes.
  ~5–7 minutes on an A100 with `max_patches=1200`.

Knobs to make it cheaper:
- `--max_patches` lower (≥200 is fine for routing statistics)
- `--max_slides_per_class` lower
- `--top_k_concepts` lower (visualization only)
