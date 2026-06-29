# Concept-Aware Wasserstein Routing with Vision-Language Guidance for Few-Shot WSI Classification

Our framework combines frozen pathology foundation models with:

- **Semantic Wasserstein Routing (SWR)** using Unbalanced Optimal Transport (UOT)
- **Barycentric Prototype Memory**
- Multi-scale pathology features
- Vision-language prompting for few-shot adaptation

---

## Repository Structure

```text
.
├── README.md
├── patch_extraction.py
├── conch_feats.py
├── plip_feats.py
│
├── train
│   ├── conch
│   │   ├── train_conch_brca.py
│   │   └── train_conch_rcc.py
│   │
│   └── plip
│       ├── train_plip_brca.py
│       └── train_plip_rcc.py
│
└── ablation
    ├── ablation.py
    │
    └── interpretability
        ├── README.md
        ├── coordinate_utils.py
        ├── extract_pi.py
        ├── visualize_concept_effects.py
        └── visualize_transport_maps.py
```

---

# Environment Setup

Set the required environment variables before running any scripts.

```bash
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES=0
```

---

# Dataset Structure

### Low-Magnification Images (5×)

```text
<wsi_root>/
└── <slide_id>/
    └── 5x/
        ├── patch1.jpg
        ├── patch2.jpg
        └── ...
```

---

### High-Magnification Features (20×)

```text
<features_root>/
└── <slide_id>.pt
```

where each tensor has shape

```text
N × D
```

---

### BRCA Labels

CSV/TSV should contain

```text
submitter_id,label
TCGA-D8-A3Z6,IDC
TCGA-BH-A0DG,ILC
```

---

# Usage

## 1. Patch Extraction

```bash
python patch_extraction.py \
    --dataset TCGA-lung \
    --slide_format svs \
    --tile_size 224 \
    --base_mag 20 \
    --magnifications 0 1 \
    --workers 8
```

---

## 2. Feature Extraction

### CONCH

```bash
python conch_feats.py \
    --input_root /path/to/images \
    --out_root /path/to/features \
    --magnification 20x \
    --batch_size 128 \
    --gpu 0
```

### PLIP

```bash
python plip_feats.py \
    --input_root /path/to/images \
    --out_root /path/to/features \
    --magnification 20x \
    --batch_size 128 \
    --gpu 0
```

---

## 3. Training

Default run

```bash
python train/conch/train_conch_brca.py
```

Custom run

```bash
python train/conch/train_conch_brca.py \
    --brca_root /data/brca \
    --conch_feats_root /data/features \
    --label_csv labels.tsv \
    --k 16 \
    --epochs 50
```

Resume training

```bash
python train/conch/train_conch_brca.py --resume
```

Resume from checkpoint

```bash
python train/conch/train_conch_brca.py \
    --resume_from checkpoint.pth
```

Rebuild splits

```bash
python train/conch/train_conch_brca.py \
    --rebuild_splits
```

---

# Ablation Studies

Run structural ablations

```bash
python ablation/ablation.py \
    --k 16 \
    --ablation_mode full_uot
```

---

# Interpretability

## Concept Analysis

```bash
python visualize_concept_effects.py \
    --checkpoint model.pth \
    --output_dir outputs/concept_effects \
    --split test
```

## Transport Maps

```bash
python visualize_transport_maps.py \
    --checkpoint model.pth \
    --output_dir outputs/transport_maps \
    --split test
```

---

# Configuration

| Component | Setting |
|-----------|---------|
| Transport | Unbalanced Optimal Transport |
| ε | 0.1 |
| τ | 1.0 |
| Sinkhorn Iterations | 50 |
| Prototype Memory | 4 prototypes per class |
| EMA Momentum | 0.999 |
| Optimizer | AdamW |
| Learning Rate | 1e−3 → 1e−5 |
| Gradient Clipping | 1.0 |
| Scheduler | Cosine Annealing |
