# Concept-Aware Wasserstein Routing with Vision-Language Guidance for Few-Shot WSI Classification

This repository contains the official implementation for a multi-scale, parameter-efficient vision-language framework designed for few-shot Weakly Supervised Classification (FSWC) of whole-slide images (WSIs). The framework integrates frozen pathology foundation backbones with **Semantic Wasserstein Routing (SWR)** using unbalanced optimal transport (UOT) and a **Barycentric Prototype Memory** bank to model complex intra-class morphological heterogeneity.

---

## 📂 Repository Structure

- ablation/
- ablation/interpretability/
- ablation/interpretability/README.md
- ablation/interpretability/coordinate_utils.py
- ablation/interpretability/extract_pi.py
- ablation/interpretability/visualize_concept_effects.py
- ablation/interpretability/visualize_transport_maps.py
- ablation/ablation.py
- train/
- train/conch/
- train/conch/train_conch_brca.py
- train/conch/train_conch_rcc.py
- train/plip/
- train/plip/train_plip_brca.py
- train/plip/train_plip_rcc.py
- README.md
- conch_feats.py
- patch_extraction.py
- plip_feats.py

---

## ⚙️ Environment & Setup

Before execution, ensure you configure the mandatory environment variables required to access the underlying vision-language foundation weights and pin compute resources:

export HF_TOKEN="your_huggingface_token_here"
export CUDA_VISIBLE_DEVICES=0

---

## 📦 Required Dataset File Structure

The network maps multi-scale hierarchies dynamically by scanning flat directory layouts (for TCGA-BRCA via explicit CSV indexing) or per-class subdirectories (for TCGA-RCC). Ensure your assets are laid out in one of the following valid structures:

### 1. Low-Magnification Input (5x Patches)
- Base Path: <wsi_root_dir>/<slide_id>/5x/*.jpg
- Alternative Path: <wsi_root_dir>/pyramid/<slide_id>/5x/*.png

### 2. High-Magnification Input (20x Features)
- Bag Path: <feats_root_dir>/<slide_id>/20x/*.pt
- Alternative Path: <feats_root_dir>/pyramid/<slide_id>/20x/*.pt
- Compressed Path: <feats_root_dir>/<slide_id>.pt (Single pre-aggregated bag tensor of shape N by D)

### 3. Label Mapping (TCGA-BRCA Specific)
For breast subtyping, slide labels must be specified via a tabular tsv or csv file matching TCGA patient barcodes. It needs a "submitter_id" column and a "label" column (with values "IDC" or "ILC").

Example:
submitter_id,label
TCGA-D8-A3Z6,IDC
TCGA-BH-A0DG,ILC

---

## 🚀 Execution & Model Training

### 1. Preprocessing & Patch Extraction
Extracts fixed-size image patches from Whole Slide Images (WSIs) using OpenSlide's DeepZoom interface. It generates patches at one or two user-specified magnification levels, filters out background tiles using an edge-based threshold, parallelized across multiple worker processes.

python patch_extraction.py \
    --dataset TCGA-lung \
    --slide_format svs \
    --tile_size 224 \
    --base_mag 20 \
    --magnifications 0 1 \
    --workers 8

### 2. Foundation Model Feature Extraction

**CONCH Feature Extraction:**
Extracts L2-normalized embeddings from 20x magnification pathology image patches organized in a TCGA-style directory structure using the pretrained MahmoodLab CONCH ViT-B-16 model. Supports GPU inference, mixed precision (AMP), and multiprocessing.

python conch_feats.py \
    --input_root /home/datasets/tcga_brca \
    --out_root /home/datasets/tcga_brca_conch_feats \
    --magnification 20x \
    --img_size 224 \
    --batch_size 128 \
    --num_workers 6 \
    --gpu 3

**PLIP Feature Extraction:**
Extracts Pathology Language-Image Pretraining (PLIP) embeddings from 20x magnification pathology patches utilizing the pretrained vinid/plip model configuration.

python plip_feats.py \
    --input_root /home/datasets/tcga_brca \
    --out_root /home/datasets/tcga_brca_plip_feats_20x \
    --magnification 20x \
    --img_size 224 \
    --batch_size 128 \
    --num_workers 6 \
    --gpu 3

### 3. Running Few-Shot Optimization Tracks
The training script manages patient-level cohort validation splits, applies class-imbalance-aware sampling routines, and initializes multi-level pathology prompting pipelines:

# Basic run with default parameters
python train/conch/train_conch_brca.py

# Custom execution overriding tracking roots and shot bounds
python train/conch/train_conch_brca.py --brca_root /data/tcga_brca/WSI --conch_feats_root /data/tcga_brca/tcga_feats --label_csv /data/tcga_brca/brca_labels.tsv --k 16 --epochs 50

### 4. Checkpoint Tracking & Re-Splitting Configurations
The framework includes full state serialization to seamlessly handle cluster preemption or fine-tuning resumption loops:

# Automatically locate and resume from the latest saved state
python train/conch/train_conch_brca.py --resume

# Resume from an explicit checkpoint path
python train/conch/train_conch_brca.py --resume_from /path/to/checkpoint.pth

# Purge current index matrices and re-split patient cohorts from scratch
python train/conch/train_conch_brca.py --rebuild_splits

---

## 🔬 Ablation Studies & Interpretability Maps

### 1. Running Structural Ablations
Runs ablation studies for the proposed multi-scale WSI classification framework by replacing the Unbalanced Optimal Transport (UOT) modules with alternative variants such as Balanced OT or No OT (cosine pooling). Trains the modified architecture and saves checkpoints and evaluation test reports.

python ablation.py \
    --k 16 \
    --ablation_mode full_uot

### 2. Reusable Interpretability Utilities
The downstream interpretability workflows are supported by the following core utility modules:
- **coordinate_utils.py:** (Utility) Handles spatial mapping of WSI patches, parses patch coordinates from pathology filenames, constructs thumbnail mosaics, and reshapes patch arrays into 2D grids for heatmaps. Cannot be run directly.
- **extract_pi.py:** (Utility) Reusable backend that loads trained checkpoints, extracts text embeddings from the prompt learner, encodes patches into features, and computes the UOT transport plans (pi) between image patches and tissue concepts. Cannot be run directly.

### 3. Dataset-Level Concept Analysis
Performs dataset-level interpretability analysis of the learned pathology concepts. Computes transport statistics over a selected split and generates publication-quality plots containing concept importance maps, concept-class transport heatmaps, and a tracking CSV.

python visualize_concept_effects.py \
    --checkpoint ./fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
    --main_module train \
    --output_dir ./interpretability_outputs/concept_effects \
    --scale 5x \
    --split test \
    --max_slides_per_class 30 \
    --max_patches 1200 \
    --top_k 8 \
    --gpu_id 0

### 4. Spatial Transport Map Projections
Generates localized spatial visualizations of the UOT plan for selected whole-slide images. Reconstructs the absolute spatial layout of patches using parsed coordinates and overlays concept-specific heatmaps on a stitched thumbnail mosaic of the slide.

python visualize_transport_maps.py \
    --checkpoint ./fewshot_ckpt_conch_rcc_k16_20x_full_seed_109/best_bal_acc_model.pth \
    --main_module train \
    --output_dir ./interpretability_outputs/transport_maps \
    --scale 5x \
    --split test \
    --num_per_class 2 \
    --top_k_concepts 5 \
    --max_patches 1500 \
    --tile_size 48 \
    --max_dim 2400 \
    --gpu_id 0 \
    --seed 0

---

## 🛠️ Configuration Details

- Semantic Wasserstein Routing (SWR): Configured via Entropically Regularized Unbalanced Optimal Transport (epsilon=0.1, tau=1.0, 50 Sinkhorn iterations, tolerance threshold 10^-4).
- Barycentric Prototype Memory: Utilizes 4 visual templates per diagnostic class optimized as Wasserstein barycenters and blended with the Euclidean mean (lambda_E=0.3), updated via EMA momentum (0.999).
- Optimization Engine: Managed via AdamW with a cosine learning rate scheduler (1e-3 base scaling down to 1e-5 floor boundary), gradient norm clipping capped at 1.0, and early stopping.

---
