"""
v3.1 — TCGA-RCC 3-class (KICH / KIRP / KIRC)
With Semantic Wasserstein Routing (SWR), CONCH encoder, RCC-specific
hierarchical prompts, automatic patient-level train/val/test splits,
class-imbalance-aware sampling, and full RESUME-FROM-CHECKPOINT support.

** v3 changes: **
- 20× patches now loaded from pre-extracted .pt feature files
  (path: <conch_feats_root>/<class>/pyramid/<class>/<slide_id>/20x/*.pt).
- Bypasses CONCH encoder for 20×; raw images still used for 5×.
- Barycentric memory uses 5× spatial tokens (config.memory_scale).

** v3.1 RESUME FIX **
- BarycentreMemory now stores `memory_initialized`, `_update_count` and
  `_batch_count` as registered buffers so they ride along inside
  state_dict() — they used to be plain Python attributes, which caused
  resume to leave the memory frozen / showing updates=0.
- Backwards-compat: `restore_initialized_flag_from_buffers()` is called
  after load_state_dict; if the checkpoint predates this fix but the
  saved barycenters are clearly populated, the flag is auto-restored.
- load_checkpoint() now calls restore_initialized_flag_from_buffers().
- main() resume block rebuilds memory only when needed (memory not
  initialized after loading checkpoint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import json
import random
import logging
import copy
import math
import gc
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score
)
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# ============================================================================
# CONCH IMPORTS
# ============================================================================
try:
    from conch.open_clip_custom import (
        create_model_from_pretrained, get_tokenizer, tokenize)
except ImportError:
    raise ImportError(
        "CONCH not installed. Install with: "
        "pip install git+https://github.com/Mahmoodlab/CONCH.git")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# Tokenizer compatibility shim
# ===========================================================================
def _safe_tokenize(texts, tokenizer, context_length=128):
    """Tokenize a list of strings into a LongTensor of input_ids."""
    if isinstance(texts, str):
        texts = [texts]
    try:
        return tokenize(texts=texts, tokenizer=tokenizer)
    except AttributeError:
        pass
    except TypeError:
        pass
    out = tokenizer(
        texts,
        max_length=context_length,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    if isinstance(out, dict):
        ids = out['input_ids']
    else:
        ids = out.input_ids
    return ids.long()


# ===========================================================================
# GPU helpers
# ===========================================================================
def grab_gpu_memory(device="cuda:0", fraction=0.05):
    if not torch.cuda.is_available():
        logger.warning("No GPU available — skipping memory lock.")
        return
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
    grab_bytes = int(free_mem * fraction)
    n_elements = grab_bytes // 4
    try:
        dummy = torch.empty(n_elements, dtype=torch.float32, device=device)
        del dummy
        cached_gb = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
        logger.info(f"[GPU LOCK] Locked {cached_gb:.1f} GB.")
    except RuntimeError as e:
        logger.error(f"[GPU LOCK] Failed: {e}")


def print_gpu_status(device="cuda:0", label=""):
    if not torch.cuda.is_available():
        return
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
    allocated = torch.cuda.memory_allocated(device_idx)
    cached = torch.cuda.memory_reserved(device_idx)
    print(f"  [GPU {label}] allocated={allocated/1024**3:.1f}GB  "
          f"cached={cached/1024**3:.1f}GB  driver_free={free_mem/1024**3:.1f}GB  "
          f"total={total_mem/1024**3:.1f}GB")


# ===========================================================================
# Configuration  (RCC 3-class)
# ===========================================================================
class Config:
    gpu_device_id = 0

    # ====== RCC PATHS ======
    rcc_root = "/home/datasets/tcga_rcc/WSI/"
    conch_feats_root = "/home/datasets/tcga_rcc/conch_feats"
    class_dir_names = {
        "KICH": "kich",
        "KIRP": "kirp",
        "KIRC": "kirc",
    }
    class_to_idx = {"KICH": 0, "KIRP": 1, "KIRC": 2}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = ["KICH", "KIRP", "KIRC"]
    num_classes = 3

    # Splits
    split_json = "./rcc_data_splits.json"
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    split_seed = 109

    # ===== MODEL: CONCH =====
    conch_model_name = "conch_ViT-B-16"
    conch_checkpoint = "hf_hub:MahmoodLab/conch"
    hf_auth_token = os.environ.get("HF_TOKEN", None)
    conch_img_size = 224
    embedding_dim = 512
    context_length = 128

    # Prompt
    num_visual_prompts = 8
    deep_prompt_depth = 4
    num_soft_tokens = 16
    num_prompts_per_class = 4
    use_nested_prompts = True
    use_patch_level_prompts = True
    use_slide_level_prompts = True
    use_multi_templates = True

    # Few-shot
    few_shot_train = True
    few_shot_k_train = 16
    few_shot_k_memory = None
    few_shot_val = False

    # Multi-scale
    use_multi_scale = True
    scales = ['5x', '20x']
    scale_max_patches = {'5x': 1000, '20x': 15000}
    scale_min_patches = {'5x': 10, '20x': 30}
    multi_scale_fusion = 'gate'
    share_encoder_across_scales = True
    use_scale_specific_prompts = True

    memory_scale = '5x'
    feature_file_template = "features_{scale}.pt"

    # Aggregation
    aggregation_method = 'wsi_attention'
    graph_scales = ['5x', '20x']
    agg_num_prompts = 8
    agg_num_heads = 8
    agg_num_layers = 2
    agg_hidden_dim = 256
    agg_dropout = None

    # Patch handling
    patch_strategy = 'cap_and_mask'
    max_patches_per_slide = 6000
    min_patches_per_slide = 20
    use_patient_level_split = True
    target_magnification = '20x'

    # UOT
    uot_tau_a = 1.0
    uot_tau_b = 1.0
    uot_epsilon = 0.1
    uot_max_iter = 50
    uot_adaptive_stop = True
    uot_convergence_thresh = 1e-4
    uot_mu = 0.5
    uot_per_tile = False

    # Barycenter memory
    num_patch_tokens = None
    vit_hidden_dim = None
    memory_num_prototypes = None
    memory_ema_momentum = None
    memory_ema_warmup_epochs = None
    memory_ema_min_momentum = None
    memory_ema_update_every = None
    memory_ema_noise_std = None
    memory_recompute_every = 0
    memory_update_during_train = True
    bary_ot_epsilon = 0.1
    bary_outer_iters = None
    bary_sinkhorn_iters = 50
    bary_euclidean_mix = None

    use_direct_patch_class = False
    use_learnable_loss_weights = False

    # GPU / batching
    batch_size = 1
    gradient_accumulation_steps = None
    use_gradient_checkpointing = True
    use_chunked_encoding = True
    encoding_chunk_size = 16
    empty_cache_frequency = 2
    pin_memory = True

    # Training
    num_epochs = 50
    learning_rate = None
    min_lr = 1e-6
    weight_decay = None
    grad_clip = None
    mixed_precision = True
    label_smoothing = None
    early_stopping_patience = None

    # Anti-overfitting
    rdrop_alpha = None
    patch_dropout_rate = 0.10
    feature_noise_std = None
    min_train_epochs = None
    swa_start_epoch = 40
    swa_enabled = False
    overfit_gap_threshold = 0.15
    train_passes_per_epoch = None

    use_stain_augmentation = True

    device = f"cuda:{gpu_device_id}" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_dir = None
    val_frequency = 3
    slide_sampler_strategy = 'class_and_sqrt_size'

    ablation_name = "rcc_3class_swr_CONCH"
    disable_uot = False
    disable_patch_prompts = False
    disable_slide_prompts = False
    disable_memory = False
    disable_multiscale = False
    disable_deep_prompts = False
    disable_graph = False
    disable_ema = False

    mixup_alpha = None
    prototype_diversity_weight = None
    prototype_contrastive_weight = None
    memory_replay_ratio = None

    # Resume
    resume = False
    resume_from = None
    rebuild_splits = False


# ===========================================================================
# Shot-Adaptive Config
# ===========================================================================
def apply_shot_adaptive_config(cfg):
    k = cfg.few_shot_k_train
    total_train = k * cfg.num_classes
    logger.info(f"\n{'='*60}\nSHOT-ADAPTIVE CONFIG: k={k}, classes={cfg.num_classes}, "
                f"total_train={total_train}\n{'='*60}")

    if cfg.memory_num_prototypes is None:
        cfg.memory_num_prototypes = min(max(int(math.sqrt(k) * 2), 2), 12)
    if cfg.few_shot_k_memory is None:
        cfg.few_shot_k_memory = k
    if cfg.memory_ema_momentum is None:
        cfg.memory_ema_momentum = min(0.999, 0.97 + 0.001 * k)
    if cfg.memory_ema_min_momentum is None:
        cfg.memory_ema_min_momentum = 0.5
    if cfg.memory_ema_warmup_epochs is None:
        cfg.memory_ema_warmup_epochs = max(10, min(25, k * 2))
    if cfg.memory_ema_update_every is None:
        cfg.memory_ema_update_every = max(3, min(10, k // 2))
    if cfg.memory_ema_noise_std is None:
        cfg.memory_ema_noise_std = max(0.002, 0.01 / math.sqrt(k / 8))
    if cfg.bary_outer_iters is None:
        avg_per_proto = max(1, k // cfg.memory_num_prototypes)
        cfg.bary_outer_iters = min(20, max(5, avg_per_proto * 3))
    if cfg.bary_euclidean_mix is None:
        cfg.bary_euclidean_mix = max(0.1, 0.4 - 0.015 * k)
    if cfg.label_smoothing is None:
        cfg.label_smoothing = max(0.1, min(0.35, 0.35 - 0.005 * k))
    if cfg.agg_dropout is None:
        cfg.agg_dropout = max(0.3, min(0.7, 0.7 - 0.01 * k))
    if cfg.weight_decay is None:
        cfg.weight_decay = max(0.05, min(0.2, 0.2 - 0.003 * k))
    if cfg.learning_rate is None:
        cfg.learning_rate = min(2e-3, 8e-4 * math.sqrt(k / 8))
    if cfg.grad_clip is None:
        cfg.grad_clip = max(0.5, 1.0 - 0.02 * k)
    if cfg.gradient_accumulation_steps is None:
        cfg.gradient_accumulation_steps = max(4, min(12, total_train))
    if cfg.num_epochs is None:
        cfg.num_epochs = max(40, min(100, int(70 + (8 - k) * 2)))
    if cfg.early_stopping_patience is None:
        cfg.early_stopping_patience = max(12, cfg.num_epochs // 3)
    if cfg.mixup_alpha is None:
        cfg.mixup_alpha = min(0.5, 0.15 + 0.015 * k)
    if cfg.prototype_diversity_weight is None:
        cfg.prototype_diversity_weight = min(0.1, 0.02 * math.sqrt(k / 8))
    if cfg.prototype_contrastive_weight is None:
        cfg.prototype_contrastive_weight = min(0.05, 0.01 * math.sqrt(k / 8))
    if cfg.memory_replay_ratio is None:
        cfg.memory_replay_ratio = max(0.0, 0.3 - 0.015 * k)
    if cfg.rdrop_alpha is None:
        cfg.rdrop_alpha = max(0.5, min(3.0, 3.0 - 0.1 * k))
    if cfg.patch_dropout_rate is None:
        cfg.patch_dropout_rate = max(0.1, min(0.4, 0.4 - 0.01 * k))
    if cfg.feature_noise_std is None:
        cfg.feature_noise_std = max(0.01, min(0.05, 0.05 - 0.001 * k))
    if cfg.min_train_epochs is None:
        cfg.min_train_epochs = max(15, cfg.num_epochs // 3)
    if cfg.swa_start_epoch is None:
        cfg.swa_start_epoch = max(20, int(cfg.num_epochs * 0.4))
    if cfg.train_passes_per_epoch is None:
        cfg.train_passes_per_epoch = max(1, min(4, 32 // total_train))
    if cfg.save_dir is None:
        cfg.save_dir = f"./fewshot_ckpt_conch_rcc_k{k}_20x_full_seed_109"

    logger.info(f"  Prototypes         : {cfg.memory_num_prototypes}")
    logger.info(f"  EMA target mom     : {cfg.memory_ema_momentum:.4f}")
    logger.info(f"  EMA warmup epochs  : {cfg.memory_ema_warmup_epochs}")
    logger.info(f"  Label smoothing    : {cfg.label_smoothing:.3f}")
    logger.info(f"  Dropout            : {cfg.agg_dropout:.3f}")
    logger.info(f"  Weight decay       : {cfg.weight_decay:.4f}")
    logger.info(f"  LR                 : {cfg.learning_rate:.6f}")
    logger.info(f"  Grad accum         : {cfg.gradient_accumulation_steps}")
    logger.info(f"  Epochs             : {cfg.num_epochs} (min={cfg.min_train_epochs})")
    logger.info(f"  R-Drop alpha       : {cfg.rdrop_alpha:.2f}")
    logger.info(f"  Save dir           : {cfg.save_dir}")
    logger.info(f"{'='*60}\n")
    return cfg


# ===========================================================================
# RCC-specific Prompt Definitions
# ===========================================================================
ARCHITECTURAL_PATTERNS = {
    "Clear cell nested architecture": {
        "description": (
            "Sheets and pseudo-alveolar nests of tumor cells with optically "
            "clear cytoplasm separated by thin fibrovascular septa, forming "
            "a delicate alveolar pattern at low power."),
        "magnification": "5x", "class_association": "KIRC", "diagnostic_weight": 1.0},
    "Prominent sinusoidal vasculature": {
        "description": (
            "Numerous arborizing thin-walled sinusoidal capillaries wrapping "
            "around tumor nests, often engorged with red blood cells."),
        "magnification": "both", "class_association": "KIRC", "diagnostic_weight": 0.95},
    "Small tumor nests with delicate septae": {
        "description": (
            "Compact round nests of clear tumor cells separated by delicate "
            "fibrovascular septae lined by flat endothelial cells, lacking "
            "fibrovascular cores or true papillae."),
        "magnification": "20x", "class_association": "KIRC", "diagnostic_weight": 0.9},
    "Solid sheet plant-like architecture": {
        "description": (
            "Broad cohesive sheets of polygonal tumor cells with pale "
            "eosinophilic cytoplasm forming a tiled or mosaic plant-cell "
            "growth pattern with minimal stromal interruption."),
        "magnification": "5x", "class_association": "KICH", "diagnostic_weight": 1.0},
    "Geographic mosaic cell grouping": {
        "description": (
            "Geographic patches and islands of polygonal tumor cells with "
            "well-defined dense cell borders and accentuated peripheral "
            "cytoplasmic outlines."),
        "magnification": "20x", "class_association": "KICH", "diagnostic_weight": 0.85},
    "Broad cell plates without fibrovascular cores": {
        "description": (
            "Expansive cohesive plates of tumor cells lacking central "
            "fibrovascular cores or papillary structures, with sparse "
            "mitotic activity."),
        "magnification": "20x", "class_association": "KICH", "diagnostic_weight": 0.85},
    "True papillary architecture": {
        "description": (
            "Well-formed papillae with fibrovascular cores lined by a single "
            "or pseudostratified layer of tumor epithelium with crowded "
            "elongated nuclei mimicking stratification."),
        "magnification": "5x", "class_association": "KIRP", "diagnostic_weight": 1.0},
    "Trabecular and pseudopapillary growth": {
        "description": (
            "Linear cords of tumor cells forming trabecular slits within "
            "loose fibrotic stroma alongside incomplete pseudopapillary "
            "areas lacking central cores."),
        "magnification": "20x", "class_association": "KIRP", "diagnostic_weight": 0.85},
    "Foamy macrophage aggregates in cores": {
        "description": (
            "Pale yellow zones of lipid-laden foamy macrophages with "
            "vacuolated cytoplasm aggregating within papillary cores or "
            "luminal spaces, often with hemosiderin pigment granules."),
        "magnification": "both", "class_association": "KIRP", "diagnostic_weight": 0.95},
    "Psammoma body formation": {
        "description": (
            "Concentric layered round calcifications visible in papillary "
            "cores or adjacent stroma, with scattered stromal mineralization."),
        "magnification": "both", "class_association": "KIRP", "diagnostic_weight": 0.9},
}

CYTOLOGICAL_FEATURES = {
    "Clear vacuolated cytoplasm with peripheral nucleus": {
        "description": (
            "Tumor cells with optically clear vacuolated cytoplasm filled "
            "with lipid and glycogen, eccentric peripheral nuclei pushed "
            "by abundant cytoplasm, and crisp well-demarcated membranes."),
        "magnification": "20x", "class_association": "KIRC", "diagnostic_weight": 1.0},
    "Low-grade round nuclei with inconspicuous nucleoli": {
        "description": (
            "Round uniform nuclei with finely dispersed evenly distributed "
            "chromatin and small or absent nucleoli, corresponding to "
            "ISUP grade 1 to 2 morphology."),
        "magnification": "20x", "class_association": "KIRC", "diagnostic_weight": 0.85},
    "Prominent perinuclear halo with reticulated cytoplasm": {
        "description": (
            "Polygonal tumor cells showing a wide clear perinuclear halo "
            "and finely reticulated microvesicular pale eosinophilic "
            "cytoplasm with eosinophilic granularity."),
        "magnification": "20x", "class_association": "KICH", "diagnostic_weight": 1.0},
    "Wrinkled raisinoid nuclei with binucleation": {
        "description": (
            "Irregular wrinkled nuclear contours with raisinoid appearance "
            "and frequent binucleation, occurring within polygonal cells "
            "with dense cytoplasmic borders."),
        "magnification": "20x", "class_association": "KICH", "diagnostic_weight": 0.95},
    "Pseudostratified crowded papillary epithelium": {
        "description": (
            "Papillae lined by tumor cells with high nuclear crowding, "
            "elongated hyperchromatic darkly stained nuclei, and apical "
            "pseudostratification with frequent mitotic figures."),
        "magnification": "20x", "class_association": "KIRP", "diagnostic_weight": 0.95},
}

SHARED_MICROENVIRONMENT = {
    "Tumor necrosis": {
        "description": (
            "Areas of coagulative necrosis within tumor showing ghost "
            "outlines of dead cells, nuclear debris, and eosinophilic coagulum."),
        "magnification": "both", "class_association": "shared", "diagnostic_weight": 0.5},
    "Tumor-associated inflammation": {
        "description": (
            "Mixed inflammatory infiltrate within and around tumor "
            "including lymphocytes, plasma cells and macrophages."),
        "magnification": "both", "class_association": "shared", "diagnostic_weight": 0.4},
    "Tumor vasculature and hemorrhage": {
        "description": (
            "Abnormal thin-walled blood vessels within tumor stroma "
            "including dilated capillaries and focal areas of hemorrhage."),
        "magnification": "20x", "class_association": "shared", "diagnostic_weight": 0.4},
    "Mitotic figures": {
        "description": (
            "Cells undergoing mitotic division with visible condensed "
            "chromosomes indicating proliferative activity."),
        "magnification": "20x", "class_association": "shared", "diagnostic_weight": 0.5},
    "Desmoplastic stroma": {
        "description": (
            "Dense fibrous connective tissue reaction surrounding tumor "
            "nests with activated fibroblasts and collagen deposition."),
        "magnification": "both", "class_association": "shared", "diagnostic_weight": 0.4},
    "Normal renal parenchyma": {
        "description": (
            "Adjacent benign renal tubules and glomeruli with intact "
            "basement membranes and unremarkable interstitium."),
        "magnification": "both", "class_association": "shared", "diagnostic_weight": 0.3},
}

ROUTING_TEMPLATES = [
    "a histopathology image showing {name}, characterized by {description}",
    "microscopic view of {name} pattern in renal tissue with {description}",
    "H&E stained section demonstrating {name} with features including {description}",
    "pathological finding of {name} in renal cell carcinoma showing {description}",
]

CLASS_KNOWLEDGE_V2 = {
    "KICH": {"primary": (
        "Chromophobe renal cell carcinoma characterized by sheets of "
        "polygonal cells with pale eosinophilic reticulated cytoplasm, "
        "perinuclear halos, raisinoid wrinkled nuclei, and frequent "
        "binucleation, growing in solid plant-like mosaic patterns "
        "without fibrovascular cores.")},
    "KIRP": {"primary": (
        "Papillary renal cell carcinoma showing true papillae with "
        "fibrovascular cores lined by pseudostratified tumor epithelium, "
        "frequent foamy macrophage aggregates within papillary cores, "
        "and characteristic psammoma body calcifications.")},
    "KIRC": {"primary": (
        "Clear cell renal cell carcinoma characterized by nests of cells "
        "with optically clear vacuolated cytoplasm rich in lipid and "
        "glycogen, separated by delicate fibrovascular septa with "
        "prominent sinusoidal capillary networks.")},
}

CLASS_TEMPLATES = [
    "a histopathological image of {name}, {description}",
    "an H&E stained slide showing {name}, {description}",
    "microscopic examination revealing {name}, {description}",
]


def build_routing_concepts(target_scale=None):
    all_concepts = {}
    all_concepts.update(ARCHITECTURAL_PATTERNS)
    all_concepts.update(CYTOLOGICAL_FEATURES)
    all_concepts.update(SHARED_MICROENVIRONMENT)
    if target_scale is not None:
        all_concepts = {n: m for n, m in all_concepts.items()
                        if m.get("magnification", "both") in ("both", target_scale)}
    return all_concepts


def build_routing_prompts(all_concepts):
    concept_names = list(all_concepts.keys())
    return concept_names, [
        ROUTING_TEMPLATES[0].format(name=n, description=all_concepts[n]["description"])
        for n in concept_names]


def build_class_prompts(class_names):
    return [
        CLASS_TEMPLATES[0].format(name=cn, description=CLASS_KNOWLEDGE_V2[cn]["primary"])
        if cn in CLASS_KNOWLEDGE_V2 else f"a histopathology image of {cn}"
        for cn in class_names]


# ===========================================================================
# Stain augmentation
# ===========================================================================
class StainAugmentation:
    def __init__(self, sigma_hed=0.05, sigma_brightness=0.1):
        self.sigma_hed = sigma_hed
        self.sigma_brightness = sigma_brightness

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        od = -np.log(img_np.clip(1e-6, 1.0))
        od *= (1 + np.random.normal(0, self.sigma_hed, (1, 1, 3)).astype(np.float32))
        od += np.random.normal(0, self.sigma_brightness * 0.01, (1, 1, 3)).astype(np.float32)
        img_np = np.exp(-od).clip(0, 1)
        return Image.fromarray((img_np * 255).astype(np.uint8))


# ===========================================================================
# UOT
# ===========================================================================
class UnbalancedOptimalTransport(nn.Module):
    def __init__(self, tau_a=1.0, tau_b=1.0, epsilon=0.1, max_iter=50,
                 adaptive_stop=True, convergence_thresh=1e-4):
        super().__init__()
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.adaptive_stop = adaptive_stop
        self.convergence_thresh = convergence_thresh

    def compute_cost_matrix(self, F_feat, G_feat):
        F_norm = F_feat / (torch.norm(F_feat, dim=-1, keepdim=True) + 1e-6)
        G_norm = G_feat / (torch.norm(G_feat, dim=-1, keepdim=True) + 1e-6)
        return 1.0 - torch.clamp(torch.bmm(F_norm, G_norm.transpose(1, 2)), -0.9999, 0.9999)

    def solve_uot_stabilized(self, C, a, b):
        B, M, N = C.shape
        device = C.device
        eps = self.epsilon + 1e-10
        log_a = torch.log(a.clamp(min=1e-12))
        log_b = torch.log(b.clamp(min=1e-12))
        log_u = torch.zeros(B, M, device=device, dtype=torch.float32)
        log_v = torch.zeros(B, N, device=device, dtype=torch.float32)
        log_K = -C / eps
        prev_lu = log_u.clone()
        for it in range(self.max_iter):
            lKv = log_K + log_v.unsqueeze(1)
            mx = lKv.max(dim=2, keepdim=True)[0]
            log_u = (self.tau_a / (self.tau_a + eps)) * (
                log_a - (torch.logsumexp(lKv - mx, dim=2) + mx.squeeze(2)))
            lKu = log_K.transpose(1, 2) + log_u.unsqueeze(1)
            mx = lKu.max(dim=2, keepdim=True)[0]
            log_v = (self.tau_b / (self.tau_b + eps)) * (
                log_b - (torch.logsumexp(lKu - mx, dim=2) + mx.squeeze(2)))
            if self.adaptive_stop and it > 5 and it % 5 == 0:
                if (log_u - prev_lu).abs().max().item() < self.convergence_thresh:
                    break
                prev_lu = log_u.clone()
        pi = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
        d = torch.sum(pi * C, dim=(1, 2))
        d = torch.where(torch.isnan(d) | torch.isinf(d),
                        torch.tensor(2.0, device=device), d)
        return torch.clamp(d, 0.0, 2.0), pi

    def forward(self, F_feat, G_feat, a=None, b=None, return_pi=False):
        with torch.amp.autocast('cuda', enabled=False):
            F_feat, G_feat = F_feat.float(), G_feat.float()
            B, M, _ = F_feat.shape
            _, N, _ = G_feat.shape
            C = self.compute_cost_matrix(F_feat, G_feat)
            if a is None:
                a = torch.ones(B, M, device=F_feat.device, dtype=torch.float32) / M
            if b is None:
                b = torch.ones(B, N, device=F_feat.device, dtype=torch.float32) / N
            a = a.float() / (a.float().sum(1, keepdim=True) + 1e-10)
            b = b.float() / (b.float().sum(1, keepdim=True) + 1e-10)
            d, pi = self.solve_uot_stabilized(C, a, b)
            return (d, pi) if return_pi else d


# ===========================================================================
# Aggregation
# ===========================================================================
class BaseAggregation(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

    def create_mask(self, num_patches, patch_counts, device):
        if patch_counts is not None:
            return torch.arange(num_patches, device=device).unsqueeze(0) < patch_counts.unsqueeze(1)
        return None


class PromptCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim), nn.Dropout(dropout))

    def forward(self, prompts, patches, mask=None):
        B, P, _ = prompts.shape
        N = patches.shape[1]
        Q = self.q_proj(prompts).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(patches).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(patches).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout_layer(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, P, self.feature_dim)
        prompts = self.norm1(prompts + self.out_proj(out))
        prompts = self.norm2(prompts + self.ffn(prompts))
        return prompts, attn.mean(dim=1)


class PromptBasedAggregation(BaseAggregation):
    def __init__(self, feature_dim=512, num_prompts=8, num_heads=8,
                 dropout=0.1, num_layers=2):
        super().__init__(feature_dim, dropout)
        self.prompt_tokens = nn.Parameter(torch.randn(1, num_prompts, feature_dim) * 0.01)
        self.cross_attention_layers = nn.ModuleList([
            PromptCrossAttention(feature_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(feature_dim, feature_dim))

    def forward(self, patch_features, patch_counts=None, return_attention=False,
                text_features=None, uot_module=None, concept_weights=None):
        B = patch_features.shape[0]
        prompts = self.prompt_tokens.expand(B, -1, -1)
        mask = self.create_mask(patch_features.shape[1], patch_counts, patch_features.device)
        for layer in self.cross_attention_layers:
            prompts, attn = layer(prompts, patch_features, mask)
        slide = self.final_proj(prompts.mean(dim=1))
        return (slide, attn.squeeze(1)) if return_attention else slide


class WSIAttentionAggregation(BaseAggregation):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super().__init__(feature_dim, dropout)
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.slide_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.01)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim), nn.Dropout(dropout))

    def forward(self, patch_features, patch_counts=None, return_attention=False,
                text_features=None, uot_module=None, concept_weights=None):
        B, N, _ = patch_features.shape
        Q = self.q_proj(self.slide_query.expand(B, -1, -1)).view(
            B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(patch_features).view(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(patch_features).view(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.create_mask(N, patch_counts, patch_features.device)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, 1, self.feature_dim)
        slide = self.output_proj(out.squeeze(1))
        return (slide, attn.mean(dim=1).squeeze(1)) if return_attention else slide


# ===========================================================================
# SWR
# ===========================================================================
class SemanticWassersteinRouting(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=512, dropout=0.1, num_concepts=None):
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        self.proj_out = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.concept_importance = (
            nn.Parameter(torch.ones(num_concepts) * 0.5)
            if num_concepts is not None else None)
        self.routing_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, patch_features, text_features, uot_module,
                patch_counts=None, concept_weights=None):
        B, N, D = patch_features.shape
        device = patch_features.device
        _, pi = uot_module(patch_features, text_features, return_pi=True)
        if concept_weights is not None:
            pi = pi * concept_weights.to(device).unsqueeze(0).unsqueeze(1)
        if self.concept_importance is not None:
            pi = pi * torch.sigmoid(self.concept_importance).to(device).unsqueeze(0).unsqueeze(1)
        A_SWR = torch.bmm(pi, pi.transpose(1, 2))
        if patch_counts is not None:
            mask = torch.arange(N, device=device).unsqueeze(0) < patch_counts.unsqueeze(1)
            A_SWR = A_SWR * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        row_sum = A_SWR.sum(dim=-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(row_sum))
        A_SWR_norm = torch.bmm(torch.bmm(D_inv_sqrt, A_SWR), D_inv_sqrt)
        strength = torch.sigmoid(self.routing_strength)
        h = self.activation(self.proj_in(patch_features))
        routed_h = self.proj_out(torch.bmm(A_SWR_norm, h))
        routed_h = self.dropout(routed_h)
        return self.norm2(self.norm1(patch_features) + strength * routed_h)


class SemanticWassersteinAggregation(BaseAggregation):
    def __init__(self, feature_dim=512, dropout=0.1, num_concepts=None):
        super().__init__(feature_dim, dropout)
        self.swr_layer = SemanticWassersteinRouting(feature_dim, feature_dim, dropout, num_concepts)
        self.readout_attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(),
            nn.Linear(feature_dim // 2, 1))
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim), nn.Dropout(dropout))

    def forward(self, patch_features, patch_counts=None, return_attention=False,
                text_features=None, uot_module=None, concept_weights=None):
        B, N, D = patch_features.shape
        h_routed = self.swr_layer(patch_features, text_features, uot_module,
                                  patch_counts, concept_weights)
        attn_scores = self.readout_attn(h_routed).squeeze(-1)
        if patch_counts is not None:
            mask = torch.arange(N, device=patch_features.device).unsqueeze(0) < patch_counts.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        slide_feature = (h_routed * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.output_proj(slide_feature)


# ===========================================================================
# RCC SWR Prompt Learner
# ===========================================================================
class RCCSWRPromptLearner(nn.Module):
    def __init__(self, conch_model, class_names, num_soft_tokens=16, config=None):
        super().__init__()
        self.conch_model = conch_model
        self.text_model = conch_model.text
        self.tokenizer = None
        self.class_names = class_names
        self.num_soft_tokens = num_soft_tokens
        self.context_length = config.context_length if config else 128

        for param in self.text_model.parameters():
            param.requires_grad = False

        ctx_dim = self.text_model.token_embedding.embedding_dim

        all_concepts = build_routing_concepts(target_scale=None)
        concept_names, routing_prompts = build_routing_prompts(all_concepts)
        self.routing_concept_names = concept_names
        self.routing_prompt_texts = routing_prompts
        self.num_routing_concepts = len(concept_names)
        self.class_associations = [
            all_concepts[n].get("class_association", "shared") for n in concept_names]

        weights = [all_concepts[n].get("diagnostic_weight", 0.5) for n in concept_names]
        cw = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("concept_weights", cw / cw.sum())

        self.routing_soft_ctx = nn.Parameter(
            torch.randn(self.num_routing_concepts, num_soft_tokens, ctx_dim,
                        dtype=torch.float32) * 0.01)

        self.slide_template_texts = build_class_prompts(class_names)
        self.slide_soft_ctx = nn.Parameter(
            torch.randn(len(class_names), num_soft_tokens, ctx_dim,
                        dtype=torch.float32) * 0.01)

        self.register_buffer('routing_input_ids', None)
        self.register_buffer('slide_input_ids', None)

        n_kich = sum(1 for a in self.class_associations if a == 'KICH')
        n_kirp = sum(1 for a in self.class_associations if a == 'KIRP')
        n_kirc = sum(1 for a in self.class_associations if a == 'KIRC')
        n_x = sum(1 for a in self.class_associations if a == 'shared')
        logger.info(
            f"RCCSWRPromptLearner: {self.num_routing_concepts} concepts "
            f"(KICH={n_kich}, KIRP={n_kirp}, KIRC={n_kirc}, Shared={n_x})")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        for attr, templates in [('routing', self.routing_prompt_texts),
                                ('slide', self.slide_template_texts)]:
            if not templates:
                continue
            ids = _safe_tokenize(templates, tokenizer,
                                 context_length=self.context_length).long()
            if ids.shape[1] < self.context_length:
                pad = torch.zeros(ids.shape[0], self.context_length - ids.shape[1], dtype=torch.long)
                ids = torch.cat([ids, pad], dim=1)
            elif ids.shape[1] > self.context_length:
                ids = ids[:, :self.context_length]
            setattr(self, f'{attr}_input_ids', ids)

    def _build_causal_mask(self, seq_len, device, dtype):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def encode_text_with_soft_prompts(self, input_ids, soft_ctx):
        device = soft_ctx.device
        input_ids = input_ids.to(device)
        text_model = self.text_model
        ctx_len = self.context_length
        all_features = []

        for i in range(soft_ctx.shape[0]):
            ids_i = input_ids[i:i+1]
            tok_emb = text_model.token_embedding(ids_i)
            D = tok_emb.shape[-1]

            eos_pos = ids_i.argmax(dim=-1).item()
            eos_pos = max(1, min(eos_pos, ctx_len - 1))

            bos = tok_emb[:, :1, :]
            tmpl = tok_emb[:, 1:eos_pos, :]
            eos = tok_emb[:, eos_pos:eos_pos+1, :]

            prompt = torch.cat([bos, soft_ctx[i:i+1], tmpl, eos], dim=1)
            cl = prompt.shape[1]
            if cl > ctx_len:
                prompt = prompt[:, :ctx_len, :]
                new_eos = ctx_len - 1
            elif cl < ctx_len:
                pad = torch.zeros(1, ctx_len - cl, D, device=device, dtype=prompt.dtype)
                prompt = torch.cat([prompt, pad], dim=1)
                new_eos = cl - 1
            else:
                new_eos = cl - 1

            pos_emb = text_model.positional_embedding[:ctx_len].to(prompt.dtype)
            x = prompt + pos_emb.unsqueeze(0)
            x = x.permute(1, 0, 2)

            if hasattr(text_model, 'attn_mask') and text_model.attn_mask is not None:
                attn_mask = text_model.attn_mask[:ctx_len, :ctx_len].to(device=device, dtype=x.dtype)
            else:
                attn_mask = self._build_causal_mask(ctx_len, device, x.dtype)

            for blk in text_model.transformer.resblocks:
                x = blk(x, attn_mask=attn_mask)

            x = x.permute(1, 0, 2)
            x = text_model.ln_final(x)
            feat = x[:, new_eos, :]

            tp = text_model.text_projection
            if isinstance(tp, nn.Parameter) or torch.is_tensor(tp):
                feat = feat @ tp
            else:
                feat = tp(feat)

            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            all_features.append(feat)
        return all_features

    def forward(self, level='both'):
        r = {}
        if level in ['patch', 'routing', 'both']:
            r['patch_features'] = self.encode_text_with_soft_prompts(
                self.routing_input_ids, self.routing_soft_ctx)
            r['tissue_names'] = self.routing_concept_names
            r['concept_weights'] = self.concept_weights
            r['class_associations'] = self.class_associations
        if level in ['slide', 'both']:
            r['slide_features'] = self.encode_text_with_soft_prompts(
                self.slide_input_ids, self.slide_soft_ctx)
            r['class_names'] = self.class_names
        return r


# ===========================================================================
# Deep Prompted CONCH Vision Transformer
# ===========================================================================
class DeepPromptedCONCHVisionTransformer(nn.Module):
    def __init__(self, conch_model, num_prompts=8, deep_depth=12,
                 use_gradient_checkpointing=False, num_scales=1,
                 use_scale_specific_prompts=False, disable_deep_prompts=False,
                 embedding_dim=512):
        super().__init__()
        self.conch_model = conch_model
        self.trunk = conch_model.visual.trunk
        self.visual_module = conch_model.visual

        for param in self.conch_model.visual.parameters():
            param.requires_grad = False

        self.num_prompts = num_prompts
        self.deep_depth = min(deep_depth, len(self.trunk.blocks))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_scales = num_scales
        self.use_scale_specific_prompts = use_scale_specific_prompts
        self.disable_deep_prompts = disable_deep_prompts
        self.embedding_dim = embedding_dim

        hd = self.trunk.embed_dim

        if not disable_deep_prompts:
            if use_scale_specific_prompts and num_scales > 1:
                self.scale_deep_prompts = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(torch.randn(num_prompts, hd, dtype=torch.float32) * 0.01)
                        for _ in range(self.deep_depth)])
                    for _ in range(num_scales)])
            else:
                self.deep_prompts = nn.ParameterList([
                    nn.Parameter(torch.randn(num_prompts, hd, dtype=torch.float32) * 0.01)
                    for _ in range(self.deep_depth)])

        self.cls_to_embed_fallback = nn.Linear(hd, embedding_dim)

    def _get_prompts(self, scale_idx=0):
        if self.disable_deep_prompts:
            return None
        if self.use_scale_specific_prompts and self.num_scales > 1:
            return self.scale_deep_prompts[scale_idx]
        return self.deep_prompts

    def _forward_block(self, block, x):
        if self.use_gradient_checkpointing and self.training:
            def fn(inp):
                return block(inp)
            return grad_checkpoint(fn, x, use_reentrant=False)
        return block(x)

    def _project_to_embed(self, tokens):
        vm = self.visual_module
        patch_only = tokens[:, 1:, :]
        try:
            pooled = vm.attn_pool_contrast(patch_only)
            if pooled.dim() == 3:
                pooled = pooled.squeeze(1)
            if hasattr(vm, 'ln_contrast'):
                pooled = vm.ln_contrast(pooled)
            if hasattr(vm, 'proj_contrast') and vm.proj_contrast is not None:
                proj = vm.proj_contrast
                if isinstance(proj, nn.Parameter) or torch.is_tensor(proj):
                    pooled = pooled @ proj
                else:
                    pooled = proj(pooled)
            return pooled
        except Exception:
            return self.cls_to_embed_fallback(tokens[:, 0, :])

    def forward(self, pixel_values, scale_idx=0):
        pixel_values = pixel_values.float()
        B = pixel_values.shape[0]
        x = self.trunk.patch_embed(pixel_values)
        D = self.trunk.embed_dim
        if x.dim() == 4:
            if x.shape[1] == D:
                x = x.flatten(2).transpose(1, 2).contiguous()
            elif x.shape[-1] == D:
                x = x.reshape(B, -1, D).contiguous()
            else:
                raise RuntimeError(f"Unexpected patch_embed shape {tuple(x.shape)}")
        elif x.dim() == 3:
            if x.shape[-1] != D:
                if x.shape[1] == D:
                    x = x.transpose(1, 2).contiguous()
                else:
                    raise RuntimeError(f"Unexpected 3D patch_embed shape {tuple(x.shape)}")
        else:
            raise RuntimeError(f"patch_embed unsupported {x.dim()}-D shape {tuple(x.shape)}")

        cls_token = self.trunk.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        if hasattr(self.trunk, 'pos_embed') and self.trunk.pos_embed is not None:
            pe = self.trunk.pos_embed
            if pe.dim() == 4:
                pe = pe.reshape(1, -1, D)
                if pe.shape[1] == x.shape[1] - 1:
                    x[:, 1:, :] = x[:, 1:, :] + pe
                else:
                    x = x + pe[:, :x.shape[1], :]
            else:
                x = x + pe[:, :x.shape[1], :]

        x = self.trunk.pos_drop(x) if hasattr(self.trunk, 'pos_drop') else x
        if hasattr(self.trunk, 'patch_drop'):
            x = self.trunk.patch_drop(x)
        if hasattr(self.trunk, 'norm_pre'):
            x = self.trunk.norm_pre(x)

        prompts = self._get_prompts(scale_idx)

        for i, block in enumerate(self.trunk.blocks):
            use_p = (not self.disable_deep_prompts
                     and i < self.deep_depth
                     and prompts is not None)
            if use_p:
                x_wp = torch.cat([
                    x[:, :1, :],
                    prompts[i].unsqueeze(0).expand(B, -1, -1),
                    x[:, 1:, :]], dim=1)
                x_wp = self._forward_block(block, x_wp)
                x = torch.cat([x_wp[:, :1, :], x_wp[:, 1 + self.num_prompts:, :]], dim=1)
            else:
                x = self._forward_block(block, x)

        if hasattr(self.trunk, 'norm'):
            x = self.trunk.norm(x)

        cls_embed_512 = self._project_to_embed(x)
        patch_tokens = x[:, 1:, :]
        return cls_embed_512, patch_tokens


# ===========================================================================
# Multi-Scale Fusion
# ===========================================================================
class MultiScaleFusion(nn.Module):
    def __init__(self, feature_dim=512, num_scales=2, fusion_type='gate', dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'gate':
            self.gate_proj = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(feature_dim, num_scales))
            self.out_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout))
        elif fusion_type == 'concat':
            self.proj = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim))

    def forward(self, scale_features):
        if len(scale_features) == 1:
            return scale_features[0]
        stacked = torch.stack(scale_features, dim=1)
        B, S, D = stacked.shape
        if self.fusion_type == 'gate':
            gates = F.softmax(self.gate_proj(stacked.view(B, S * D)), dim=-1)
            return self.out_proj((stacked * gates.unsqueeze(-1)).sum(dim=1))
        elif self.fusion_type == 'concat':
            return self.proj(stacked.view(B, S * D))
        return stacked.mean(dim=1)


# ===========================================================================
# Sinkhorn Barycenter
# ===========================================================================
class SinkhornBarycenter:
    @staticmethod
    @torch.no_grad()
    def _sinkhorn_plan(C, epsilon=0.05, max_iter=50, convergence_thresh=1e-4):
        P = C.shape[0]
        K = torch.exp(-C / (epsilon + 1e-10))
        a = torch.ones(P, device=C.device, dtype=torch.float32) / P
        b = torch.ones(P, device=C.device, dtype=torch.float32) / P
        u = torch.ones(P, device=C.device, dtype=torch.float32)
        for it in range(max_iter):
            u_prev = u.clone()
            v = b / (K.T @ u + 1e-12)
            u = a / (K @ v + 1e-12)
            if it > 5 and it % 5 == 0:
                if (u - u_prev).abs().max() < convergence_thresh:
                    break
        return torch.diag(u) @ K @ torch.diag(v)

    @staticmethod
    @torch.no_grad()
    def compute_cost_matrix(X, Y):
        return 1.0 - (X @ Y.T).clamp(-0.9999, 0.9999)

    @classmethod
    @torch.no_grad()
    def compute_barycenter(cls, exemplars, weights=None, epsilon=0.05,
                           outer_iters=10, sinkhorn_iters=50):
        K, P, D = exemplars.shape
        device = exemplars.device
        if weights is None:
            weights = torch.ones(K, device=device, dtype=torch.float32) / K
        else:
            weights = weights / (weights.sum() + 1e-10)
        eucl_mean = F.normalize(exemplars.mean(dim=0), dim=-1)
        init_costs = torch.tensor([
            cls.compute_cost_matrix(eucl_mean, exemplars[k]).sum().item()
            for k in range(K)])
        bary = exemplars[init_costs.argmin().item()].clone()
        for _ in range(outer_iters):
            ds = torch.zeros(P, D, device=device, dtype=torch.float32)
            ws = torch.zeros(P, 1, device=device, dtype=torch.float32)
            for k in range(K):
                C = cls.compute_cost_matrix(bary, exemplars[k])
                pi = cls._sinkhorn_plan(C, epsilon=epsilon, max_iter=sinkhorn_iters)
                prs = pi.sum(dim=1, keepdim=True).clamp(min=1e-12)
                ds += weights[k] * (pi @ exemplars[k]) / prs
                ws += weights[k]
            bary = F.normalize(ds / ws.clamp(min=1e-10), dim=-1)
        return bary

    @classmethod
    @torch.no_grad()
    def compute_pairwise_ot_distances(cls, exemplars, epsilon=0.05, sinkhorn_iters=30):
        K = exemplars.shape[0]
        dist_matrix = torch.zeros(K, K, device=exemplars.device, dtype=torch.float32)
        for i in range(K):
            for j in range(i + 1, K):
                C = cls.compute_cost_matrix(exemplars[i], exemplars[j])
                pi = cls._sinkhorn_plan(C, epsilon=epsilon, max_iter=sinkhorn_iters)
                d = (pi * C).sum()
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix


# ===========================================================================
# BARYCENTER MEMORY  — v3.1 RESUME-SAFE
# ---------------------------------------------------------------------------
# ROOT CAUSE of the "FROZEN / updates=0" symptom after --resume:
#   memory_initialized, _update_count, _batch_count were plain Python
#   attributes. PyTorch's state_dict() does NOT serialize those, so after
#   load_state_dict() they reverted to __init__ defaults (False / 0 / 0).
#   With memory_initialized=False the whole memory branch was silently
#   skipped in forward_wsi and ema_update() returned immediately.
#
# FIX: the three state variables are now registered buffers (1-element
#   tensors). Python properties wrap them so the rest of the code can
#   keep using `self.memory_initialized = True` / `self._update_count += 1`
#   syntax unchanged. restore_initialized_flag_from_buffers() provides a
#   backwards-compat fallback for checkpoints saved before this fix.
# ===========================================================================
class BarycentreMemory(nn.Module):
    def __init__(self, k_shots=16, num_classes=3, num_patch_tokens=196,
                 vit_hidden_dim=768, num_prototypes=4, ema_momentum=0.99,
                 ema_warmup_epochs=10, ot_epsilon=0.1, bary_outer_iters=10,
                 bary_sinkhorn_iters=50, ema_min_momentum=0.95,
                 ema_update_every=4, ema_noise_std=0.01, euclidean_mix=0.3):
        super().__init__()
        self.k_shots = k_shots
        self.num_classes = num_classes
        self.num_patch_tokens = num_patch_tokens
        self.vit_hidden_dim = vit_hidden_dim
        self.num_prototypes = num_prototypes
        self.base_momentum = ema_momentum
        self.ema_warmup_epochs = ema_warmup_epochs
        self.ema_min_momentum = ema_min_momentum
        self.ema_update_every = ema_update_every
        self.ema_noise_std = ema_noise_std
        self.euclidean_mix = euclidean_mix
        self.current_epoch = 0          # set externally each epoch; transient, not saved
        self.ot_epsilon = ot_epsilon
        self.bary_outer_iters = bary_outer_iters
        self.bary_sinkhorn_iters = bary_sinkhorn_iters

        # ---- Persistent model state (saved in state_dict) ----
        self.register_buffer("barycenters", torch.zeros(
            num_classes, num_prototypes, num_patch_tokens, vit_hidden_dim, dtype=torch.float32))
        self.register_buffer("prototype_weights", torch.ones(
            num_classes, num_prototypes, dtype=torch.float32) / num_prototypes)
        self.register_buffer("update_magnitudes", torch.zeros(
            num_classes, num_prototypes, dtype=torch.float32))
        self.register_buffer("total_updates_per_proto", torch.zeros(
            num_classes, num_prototypes, dtype=torch.long))
        self.register_buffer("initial_barycenters", torch.zeros(
            num_classes, num_prototypes, num_patch_tokens, vit_hidden_dim, dtype=torch.float32))

        # ---- v3.1 FIX: persist memory-state flags as buffers ----
        # Using 1-element long tensors; wrapped by @property below so
        # existing `self.memory_initialized = True` / `+= 1` code works.
        self.register_buffer("_initialized_flag",  torch.zeros(1, dtype=torch.long))
        self.register_buffer("_update_count_buf",  torch.zeros(1, dtype=torch.long))
        self.register_buffer("_batch_count_buf",   torch.zeros(1, dtype=torch.long))

        # ---- Transient rolling tile buffer — NOT saved ----
        self._accumulation_buffer = {c: [] for c in range(num_classes)}
        self._accumulation_limit  = max(4, k_shots // 2)

    # ------------------------------------------------------------------
    # Property shims — let the rest of the code use the same syntax as
    # before while the underlying storage is a registered buffer.
    # ------------------------------------------------------------------
    @property
    def memory_initialized(self) -> bool:
        return bool(self._initialized_flag.item())

    @memory_initialized.setter
    def memory_initialized(self, val: bool):
        self._initialized_flag.fill_(1 if val else 0)

    @property
    def _update_count(self) -> int:
        return int(self._update_count_buf.item())

    @_update_count.setter
    def _update_count(self, val: int):
        self._update_count_buf.fill_(int(val))

    @property
    def _batch_count(self) -> int:
        return int(self._batch_count_buf.item())

    @_batch_count.setter
    def _batch_count(self, val: int):
        self._batch_count_buf.fill_(int(val))

    # ------------------------------------------------------------------
    # Backwards-compat helper: if a pre-v3.1 checkpoint is loaded, the
    # new buffers (_initialized_flag etc.) will be absent from the saved
    # state_dict and default to 0 after load_state_dict(strict=False).
    # If the barycenters are clearly populated, restore the flag here.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def restore_initialized_flag_from_buffers(self):
        """Call once after load_state_dict to fix up pre-v3.1 checkpoints."""
        if not self.memory_initialized and self.barycenters.abs().sum() > 1e-3:
            self.memory_initialized = True
            logger.info("[Memory] Backwards-compat restore: barycenters are populated "
                        "→ setting memory_initialized=True")

    # ------------------------------------------------------------------
    def get_momentum(self):
        if self.current_epoch >= self.ema_warmup_epochs:
            return self.base_momentum
        p = self.current_epoch / max(self.ema_warmup_epochs, 1)
        if p < 0.3:
            s, e, lp = 0.50, 0.70, p / 0.3
        elif p < 0.7:
            s, e, lp = 0.70, 0.90, (p - 0.3) / 0.4
        else:
            s, e, lp = 0.90, self.base_momentum, (p - 0.7) / 0.3
        return s + (e - s) * (1 - math.cos(math.pi * lp)) / 2

    @torch.no_grad()
    def _balanced_assignments(self, dist_matrix, num_prototypes):
        K = dist_matrix.shape[0]
        medoids = [torch.argmin(dist_matrix.sum(dim=1)).item()]
        for _ in range(1, num_prototypes):
            md = dist_matrix[:, medoids].min(dim=1)[0]
            medoids.append(torch.argmax(md).item())
        assignments = torch.argmin(dist_matrix[:, medoids], dim=1)
        target_size = K // num_prototypes
        remainder = K % num_prototypes
        target_sizes = [target_size + 1 if i < remainder else target_size
                        for i in range(num_prototypes)]
        for _ in range(20):
            sizes = [(assignments == i).sum().item() for i in range(num_prototypes)]
            overfull  = [i for i, sz in enumerate(sizes) if sz > target_sizes[i]]
            underfull = [i for i, sz in enumerate(sizes) if sz < target_sizes[i]]
            if not overfull or not underfull:
                break
            moved = False
            for o in overfull:
                o_idx = (assignments == o).nonzero(as_tuple=True)[0]
                if len(o_idx) == 0:
                    continue
                s_idx = torch.argsort(dist_matrix[o_idx, medoids[o]], descending=True)
                for idx in s_idx:
                    if sizes[o] <= target_sizes[o]:
                        break
                    ex = o_idx[idx]
                    best_u, best_d = None, float('inf')
                    for u in underfull:
                        d = dist_matrix[ex, medoids[u]].item()
                        if d < best_d:
                            best_d, best_u = d, u
                    if best_u is not None:
                        assignments[ex] = best_u
                        sizes[o] -= 1
                        sizes[best_u] += 1
                        moved = True
                        overfull  = [i for i, sz in enumerate(sizes) if sz > target_sizes[i]]
                        underfull = [i for i, sz in enumerate(sizes) if sz < target_sizes[i]]
                        if not overfull or not underfull:
                            break
                if not overfull or not underfull:
                    break
            if not moved:
                break
        return assignments

    @torch.no_grad()
    def _resize_buffers_if_needed(self, n_tokens, hidden_dim, device):
        if (n_tokens != self.num_patch_tokens) or (hidden_dim != self.vit_hidden_dim):
            logger.info(
                f"[Memory] Resizing buffers: "
                f"({self.num_classes},{self.num_prototypes},"
                f"{self.num_patch_tokens},{self.vit_hidden_dim}) → "
                f"({self.num_classes},{self.num_prototypes},{n_tokens},{hidden_dim})")
            self.num_patch_tokens = n_tokens
            self.vit_hidden_dim   = hidden_dim
            self.barycenters          = torch.zeros(
                self.num_classes, self.num_prototypes, n_tokens, hidden_dim,
                dtype=torch.float32, device=device)
            self.initial_barycenters  = torch.zeros_like(self.barycenters)

    @torch.no_grad()
    def init_memory(self, visual_encoder, dataloader, device, scale_idx=0):
        print(f"Initializing Barycenter Memory ({self.k_shots} shots, "
              f"{self.num_prototypes} prototypes per class)...")
        visual_encoder.eval()
        class_patches  = [[] for _ in range(self.num_classes)]
        max_collect    = max(self.k_shots * self.num_prototypes, self.k_shots * 4)
        detected_shape = None

        for images, labels in tqdm(dataloader, desc="Collecting tiles"):
            images = images.to(device=device, dtype=torch.float32)
            _, spatial_tokens = visual_encoder(images, scale_idx=scale_idx)
            if detected_shape is None:
                detected_shape = (spatial_tokens.shape[1], spatial_tokens.shape[2])
                self._resize_buffers_if_needed(
                    detected_shape[0], detected_shape[1],
                    self.barycenters.device if self.barycenters.numel() > 0 else device)
            spatial_norm = F.normalize(spatial_tokens, dim=-1)
            for i in range(images.shape[0]):
                lbl = labels[i].item()
                if lbl < self.num_classes and len(class_patches[lbl]) < max_collect:
                    class_patches[lbl].append(spatial_norm[i].cpu())
            if all(len(c) >= max_collect for c in class_patches):
                break

        for c in range(self.num_classes):
            if not class_patches[c]:
                self.barycenters[c] = F.normalize(
                    torch.randn(self.num_prototypes, self.num_patch_tokens,
                                self.vit_hidden_dim) * 0.01, dim=-1).to(self.barycenters.device)
                print(f"  Class {c}: WARNING — no exemplars, random init")
                continue
            while len(class_patches[c]) < self.num_prototypes:
                class_patches[c].append(class_patches[c][0])
            stacked = torch.stack(class_patches[c])
            if len(class_patches[c]) >= self.num_prototypes * 2:
                dm   = SinkhornBarycenter.compute_pairwise_ot_distances(
                    stacked, epsilon=self.ot_epsilon,
                    sinkhorn_iters=min(30, self.bary_sinkhorn_iters))
                asgn = self._balanced_assignments(dm, self.num_prototypes)
                cluster_members = [[] for _ in range(self.num_prototypes)]
                for idx, a in enumerate(asgn):
                    cluster_members[a.item()].append(idx)
                for ki in range(self.num_prototypes):
                    members = cluster_members[ki]
                    if not members:
                        self.barycenters[c, ki] = F.normalize(
                            stacked.mean(dim=0), dim=-1).to(self.barycenters.device)
                        self.prototype_weights[c, ki] = 1.0 / self.num_prototypes
                        continue
                    ce  = stacked[members]
                    w_b = SinkhornBarycenter.compute_barycenter(
                        ce, epsilon=self.ot_epsilon, outer_iters=self.bary_outer_iters,
                        sinkhorn_iters=self.bary_sinkhorn_iters)
                    e_b = F.normalize(ce.mean(dim=0), dim=-1)
                    mix = self.euclidean_mix
                    self.barycenters[c, ki] = F.normalize(
                        (1 - mix) * w_b + mix * e_b, dim=-1).to(self.barycenters.device)
                    self.prototype_weights[c, ki] = len(members) / float(stacked.shape[0])
                print(f"  Class {c}: {self.num_prototypes} balanced prototypes from "
                      f"{len(class_patches[c])} exemplars "
                      f"(sizes: {[len(m) for m in cluster_members]})")
            else:
                w_b  = SinkhornBarycenter.compute_barycenter(
                    stacked, epsilon=self.ot_epsilon, outer_iters=self.bary_outer_iters,
                    sinkhorn_iters=self.bary_sinkhorn_iters)
                e_b  = F.normalize(stacked.mean(dim=0), dim=-1)
                bary = F.normalize(
                    (1 - self.euclidean_mix) * w_b + self.euclidean_mix * e_b, dim=-1)
                for ki in range(self.num_prototypes):
                    self.barycenters[c, ki]         = bary.to(self.barycenters.device)
                    self.prototype_weights[c, ki]   = 1.0 / self.num_prototypes
                print(f"  Class {c}: single barycenter from {len(class_patches[c])} exemplars")

        self.prototype_weights = self.prototype_weights / (
            self.prototype_weights.sum(dim=1, keepdim=True) + 1e-10)
        self.initial_barycenters.copy_(self.barycenters)
        # --- set via properties so the buffers are written ---
        self.memory_initialized = True
        self._update_count      = 0
        self._batch_count       = 0
        self._accumulation_buffer = {c: [] for c in range(self.num_classes)}
        print(f"Memory initialised. Shape: ({self.num_classes}, {self.num_prototypes}, "
              f"{self.num_patch_tokens}, {self.vit_hidden_dim})")

    @torch.no_grad()
    def ema_update(self, spatial_tokens, labels):
        if not self.memory_initialized:
            return
        # Using property setter arithmetic: getter returns int, setter takes int
        self._batch_count = self._batch_count + 1
        spatial_norm = F.normalize(spatial_tokens.float(), dim=-1)
        for i in range(spatial_tokens.shape[0]):
            lbl = labels[i].item()
            if lbl < self.num_classes:
                self._accumulation_buffer[lbl].append(spatial_norm[i].cpu())
                max_buf = self._accumulation_limit * 2
                if len(self._accumulation_buffer[lbl]) > max_buf:
                    self._accumulation_buffer[lbl] = self._accumulation_buffer[lbl][-max_buf:]
        if self._batch_count % self.ema_update_every != 0:
            return
        momentum = self.get_momentum()
        for c in range(self.num_classes):
            buf = self._accumulation_buffer[c]
            if len(buf) < 2:
                continue
            accumulated  = torch.stack(buf).to(self.barycenters.device)
            mean_repr    = F.normalize(accumulated.mean(dim=0), dim=-1)
            proto_distances = []
            for ki in range(self.num_prototypes):
                bk = self.barycenters[c, ki]
                if bk.norm() < 1e-6:
                    proto_distances.append(torch.tensor(float('inf')))
                    continue
                cos_sim = F.cosine_similarity(bk.view(1, -1), mean_repr.view(1, -1)).item()
                proto_distances.append(torch.tensor(1.0 - cos_sim))
            proto_distances = torch.stack(proto_distances)
            temperature  = max(0.05,
                               0.5 * (1.0 - self.current_epoch / max(self.ema_warmup_epochs * 2, 1)))
            soft_weights = F.softmax(-proto_distances / temperature, dim=0)
            for ki in range(self.num_prototypes):
                if self.barycenters[c, ki].norm() < 1e-6:
                    continue
                w = soft_weights[ki].item()
                if w < 0.05:
                    continue
                old_bary = self.barycenters[c, ki]
                try:
                    C_mat       = SinkhornBarycenter.compute_cost_matrix(old_bary, mean_repr)
                    pi          = SinkhornBarycenter._sinkhorn_plan(
                        C_mat, epsilon=self.ot_epsilon * 2.0, max_iter=30)
                    prs         = pi.sum(dim=1, keepdim=True).clamp(min=1e-12)
                    aligned_new = F.normalize((pi @ mean_repr) / prs, dim=-1)
                except Exception:
                    aligned_new = mean_repr
                proto_momentum = min(momentum + (1 - momentum) * (1 - w), 0.999)
                updated        = proto_momentum * old_bary + (1 - proto_momentum) * aligned_new
                updated        = F.normalize(updated, dim=-1)
                if self.ema_noise_std > 0:
                    noise   = torch.randn_like(updated) * self.ema_noise_std * (1 - w)
                    updated = F.normalize(updated + noise, dim=-1)
                update_mag = (updated - old_bary).norm().item()
                self.update_magnitudes[c, ki] = (
                    0.9 * self.update_magnitudes[c, ki] + 0.1 * update_mag)
                self.total_updates_per_proto[c, ki] += 1
                self.barycenters[c, ki] = updated
            keep = min(2, len(buf))
            self._accumulation_buffer[c] = buf[-keep:]
        self._update_count = self._update_count + 1

    @torch.no_grad()
    def get_diagnostics(self):
        if not self.memory_initialized:
            return {}
        diag = {
            'momentum':      self.get_momentum(),
            'total_updates': self._update_count,
            'batch_count':   self._batch_count,
        }
        for c in range(self.num_classes):
            mags   = [self.update_magnitudes[c, ki].item() for ki in range(self.num_prototypes)]
            drifts = []
            for ki in range(self.num_prototypes):
                if self.initial_barycenters[c, ki].norm() > 1e-6:
                    drifts.append(
                        (self.barycenters[c, ki] - self.initial_barycenters[c, ki]).norm().item())
            diag[f'c{c}_avg_update_mag'] = float(np.mean(mags))  if mags   else 0.0
            diag[f'c{c}_avg_drift']      = float(np.mean(drifts)) if drifts else 0.0
            protos = F.normalize(self.barycenters[c].view(self.num_prototypes, -1), dim=-1)
            sim    = torch.mm(protos, protos.T)
            mask   = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=sim.device)
            if mask.sum() > 0:
                diag[f'c{c}_inter_proto_sim'] = sim[mask].mean().item()
                diag[f'c{c}_max_proto_sim']   = sim[mask].max().item()
            diag[f'c{c}_buf_size'] = len(self._accumulation_buffer.get(c, []))
        return diag

    def compute_prototype_diversity_loss(self):
        if not self.memory_initialized:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=self.barycenters.device)
        for c in range(self.num_classes):
            protos = F.normalize(self.barycenters[c].view(self.num_prototypes, -1), dim=-1)
            sim    = torch.mm(protos, protos.T)
            mask   = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=sim.device)
            total  = total + F.relu(sim[mask] - 0.5).mean()
        return total


# ===========================================================================
# RCC Concept-to-Class Mapper
# ===========================================================================
class RCCConceptToClassMapper(nn.Module):
    def __init__(self, num_concepts, num_classes=3, class_associations=None,
                 hidden_dim=256, dropout=0.1, class_to_idx=None):
        super().__init__()
        self.num_classes    = num_classes
        self.concept_weights = nn.Parameter(torch.zeros(num_concepts, num_classes))

        if class_to_idx is None:
            class_to_idx = {"KICH": 0, "KIRP": 1, "KIRC": 2}

        if class_associations is not None:
            with torch.no_grad():
                pos, neg = 0.3, -0.1
                for i, assoc in enumerate(class_associations):
                    if assoc in class_to_idx:
                        for cn, ci in class_to_idx.items():
                            self.concept_weights[i, ci] = pos if cn == assoc else neg

        self.attention_mapper = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.prior_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, concept_activations):
        prior_logits   = torch.matmul(concept_activations, self.concept_weights)
        learned_logits = self.attention_mapper(concept_activations)
        gate           = torch.sigmoid(self.prior_gate)
        combined       = gate * prior_logits + (1 - gate) * learned_logits
        probs          = F.softmax(combined, dim=-1)
        return torch.clamp(1 - probs, 0.0, 2.0)


# ===========================================================================
# Direct Patch-Class
# ===========================================================================
class DirectPatchClassPathway(nn.Module):
    def __init__(self, feature_dim=512, num_classes=3, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.patch_to_class = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.attn_gate = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

    def forward(self, patch_features, patch_counts=None):
        B, N, _ = patch_features.shape
        gates    = self.attn_gate(patch_features)
        if patch_counts is not None:
            mask  = (torch.arange(N, device=patch_features.device).unsqueeze(0)
                     < patch_counts.unsqueeze(1))
            gates = gates * mask.unsqueeze(-1).float()
        gates  = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        pooled = (patch_features * gates).sum(dim=1)
        probs  = F.softmax(self.patch_to_class(pooled), dim=-1)
        return torch.clamp(1 - probs, 0.0, 2.0)


class LearnableLossWeights(nn.Module):
    def __init__(self, num_losses=4, clamp_range=4.0):
        super().__init__()
        self.log_vars    = nn.Parameter(torch.zeros(num_losses))
        self.clamp_range = clamp_range

    def forward(self, losses):
        total, weights = torch.tensor(0.0, device=losses[0].device), []
        for i, loss in enumerate(losses):
            clv       = self.log_vars[i].clamp(-self.clamp_range, self.clamp_range)
            precision = torch.exp(-clv)
            total     = total + precision * loss + clv
            weights.append(precision.item())
        return total, weights


# ===========================================================================
# SWA Helper
# ===========================================================================
class SWAModel:
    def __init__(self, model):
        self.model_state = {}
        self.n_averaged  = 0
        for k, v in model.state_dict().items():
            self.model_state[k] = v.clone().float()

    def update(self, model):
        self.n_averaged += 1
        n = self.n_averaged
        for k, v in model.state_dict().items():
            if k in self.model_state:
                self.model_state[k] = (self.model_state[k] * (n - 1) / n + v.float() / n)

    def apply(self, model):
        model.load_state_dict(self.model_state, strict=False)
        return model


# ===========================================================================
# MAIN MODEL – MultiScaleMUOT_CONCH
# ===========================================================================
class MultiScaleMUOT_CONCH(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.device    = config.device
        self.num_scales = len(config.scales)

        print(f"Loading CONCH ({config.conch_model_name})...")
        token  = config.hf_auth_token
        img_sz = int(getattr(config, 'conch_img_size', 224))
        try:
            conch_model, self.conch_preprocess = create_model_from_pretrained(
                model_cfg=config.conch_model_name,
                checkpoint_path=config.conch_checkpoint,
                hf_auth_token=token, force_image_size=img_sz)
        except TypeError:
            conch_model, self.conch_preprocess = create_model_from_pretrained(
                model_cfg=config.conch_model_name,
                checkpoint_path=config.conch_checkpoint,
                hf_auth_token=token)
        conch_model = conch_model.float().eval()
        self.conch_tokenizer = get_tokenizer()
        self.conch_img_size  = img_sz

        trunk = conch_model.visual.trunk
        if config.num_patch_tokens is None:
            config.num_patch_tokens = (img_sz // 16) ** 2
        if config.vit_hidden_dim is None:
            config.vit_hidden_dim = trunk.embed_dim
        print(f"CONCH ViT: {config.num_patch_tokens} spatial tokens, "
              f"hidden_dim={config.vit_hidden_dim}, embed_dim={config.embedding_dim}")

        self.visual_encoder = DeepPromptedCONCHVisionTransformer(
            conch_model, num_prompts=config.num_visual_prompts,
            deep_depth=config.deep_prompt_depth,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            num_scales=self.num_scales,
            use_scale_specific_prompts=config.use_scale_specific_prompts,
            disable_deep_prompts=config.disable_deep_prompts,
            embedding_dim=config.embedding_dim)

        all_concepts         = build_routing_concepts()
        num_routing_concepts = len(all_concepts)
        class_associations   = [
            all_concepts[n].get("class_association", "shared") for n in all_concepts]

        self.scale_names       = config.scales
        self.memory_scale_name = config.memory_scale
        self.memory_scale_idx  = self.scale_names.index(self.memory_scale_name)

        self.scale_aggregators = nn.ModuleList()
        self._swr_scale_flags  = []
        for s_idx, sn in enumerate(self.scale_names):
            use_swr = (config.aggregation_method in ('graph', 'graph_attention', 'wsi_attention')
                       and sn in config.graph_scales and not config.disable_graph)
            self._swr_scale_flags.append(use_swr)
            if use_swr:
                agg = SemanticWassersteinAggregation(
                    config.embedding_dim, config.agg_dropout, num_routing_concepts)
            elif config.aggregation_method == 'prompt':
                agg = PromptBasedAggregation(
                    config.embedding_dim, config.agg_num_prompts, config.agg_num_heads,
                    config.agg_dropout, config.agg_num_layers)
            else:
                agg = WSIAttentionAggregation(
                    config.embedding_dim, config.agg_num_heads, config.agg_dropout)
            self.scale_aggregators.append(agg)

        self.scale_fusion = MultiScaleFusion(
            config.embedding_dim, self.num_scales,
            config.multi_scale_fusion, config.agg_dropout)

        self._use_pl = config.use_patch_level_prompts and not config.disable_patch_prompts
        self._use_sl = config.use_slide_level_prompts and not config.disable_slide_prompts

        self.nested_prompt_learner = RCCSWRPromptLearner(
            conch_model, config.class_names, config.num_soft_tokens, config)
        self.nested_prompt_learner.set_tokenizer(self.conch_tokenizer)

        if self._use_pl:
            self.tissue_mapper = RCCConceptToClassMapper(
                num_routing_concepts, config.num_classes, class_associations,
                config.agg_hidden_dim, config.agg_dropout,
                class_to_idx=config.class_to_idx)

        if config.use_direct_patch_class:
            self.direct_patch_class = DirectPatchClassPathway(
                config.embedding_dim, config.num_classes,
                config.agg_hidden_dim, config.agg_dropout)
            self.direct_weight = nn.Parameter(torch.tensor(0.0))

        self.uot_intra = UnbalancedOptimalTransport(
            config.uot_tau_a, config.uot_tau_b, config.uot_epsilon,
            config.uot_max_iter, config.uot_adaptive_stop, config.uot_convergence_thresh)

        if not config.disable_memory:
            self.memory = BarycentreMemory(
                config.few_shot_k_memory, config.num_classes,
                config.num_patch_tokens, config.vit_hidden_dim,
                config.memory_num_prototypes, config.memory_ema_momentum,
                config.memory_ema_warmup_epochs, config.bary_ot_epsilon,
                config.bary_outer_iters, config.bary_sinkhorn_iters,
                config.memory_ema_min_momentum, config.memory_ema_update_every,
                config.memory_ema_noise_std, config.bary_euclidean_mix)
        else:
            self.memory = None

        self.tau         = nn.Parameter(torch.tensor(0.07, dtype=torch.float32))
        self.mu_slide    = nn.Parameter(torch.tensor(config.uot_mu, dtype=torch.float32))
        self.alpha       = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.loss_weighter = (LearnableLossWeights(num_losses=5)
                              if config.use_learnable_loss_weights else None)

        self.conch_model = conch_model
        for p in self.conch_model.parameters():
            p.requires_grad = False

    def encode_image_chunked(self, pixel_values, chunk_size=128, scale_idx=0,
                             return_raw_spatial=False):
        all_global  = []
        all_spatial = [] if return_raw_spatial else None
        for i in range(0, pixel_values.shape[0], chunk_size):
            chunk = pixel_values[i:i + chunk_size]
            cls_embed_512, spatial_tok = self.visual_encoder(chunk, scale_idx=scale_idx)
            all_global.append(cls_embed_512)
            if return_raw_spatial:
                all_spatial.append(spatial_tok)
            del spatial_tok
        return (torch.cat(all_global, 0),
                torch.cat(all_spatial, 0) if return_raw_spatial else None)

    def init_memory(self, dataloader):
        if self.memory is not None:
            self.memory.init_memory(
                self.visual_encoder, dataloader, self.device,
                scale_idx=self.memory_scale_idx)

    def set_epoch(self, epoch):
        if self.memory is not None:
            self.memory.current_epoch = epoch

    def compute_probability_from_distance(self, d):
        tau    = torch.abs(self.tau).clamp(0.05, 0.5)
        logits = torch.clamp((1 - torch.clamp(d, 0.0, 2.0)) / tau, -50.0, 50.0)
        probs  = F.softmax(logits, dim=-1) + 1e-8
        return probs / probs.sum(dim=-1, keepdim=True)

    def _apply_patch_dropout(self, patch_features, patch_counts, drop_rate):
        if not self.training or drop_rate <= 0:
            return patch_features, patch_counts
        B, N, D = patch_features.shape
        device   = patch_features.device
        keep_mask = torch.rand(B, N, device=device) > drop_rate
        if patch_counts is not None:
            valid_mask = torch.arange(N, device=device).unsqueeze(0) < patch_counts.unsqueeze(1)
            keep_mask  = keep_mask & valid_mask
            for b in range(B):
                if keep_mask[b].sum() < 10:
                    keep_mask[b] = valid_mask[b]
        patch_features = patch_features * keep_mask.unsqueeze(-1).float()
        new_counts     = keep_mask.sum(dim=1).long() if patch_counts is not None else patch_counts
        return patch_features, new_counts

    def _apply_feature_noise(self, features, noise_std):
        if self.training and noise_std > 0:
            return features + torch.randn_like(features) * noise_std
        return features

    def _encode_scale(self, patches, patch_counts, scale_idx, return_spatial=False,
                      text_features=None, concept_weights=None):
        # Pre-extracted feature tensors (N, D)
        if (isinstance(patches, torch.Tensor)
                and patches.dim() == 3
                and patches.shape[-1] == self.config.embedding_dim):
            B, N, D    = patches.shape
            patch_global = F.normalize(patches, dim=-1)
            patch_global = self._apply_feature_noise(patch_global, self.config.feature_noise_std)
            if patch_counts is None:
                patch_counts = torch.full((B,), N, dtype=torch.long, device=patches.device)
            patch_global, patch_counts = self._apply_patch_dropout(
                patch_global, patch_counts, self.config.patch_dropout_rate)
            slide_feat = F.normalize(self.scale_aggregators[scale_idx](
                patch_global, patch_counts=patch_counts, text_features=text_features,
                uot_module=self.uot_intra, concept_weights=concept_weights), dim=-1)
            return slide_feat, patch_global, None, None, patch_counts

        # Raw image path (B, N, C, H, W)
        B, N, C, H, W = patches.shape
        patches_flat   = patches.view(-1, C, H, W)
        patch_global, raw_spatial = self.encode_image_chunked(
            patches_flat, self.config.encoding_chunk_size, scale_idx,
            return_raw_spatial=return_spatial)
        del patches_flat
        patch_global = F.normalize(patch_global.view(B, N, -1), dim=-1)
        patch_global = self._apply_feature_noise(patch_global, self.config.feature_noise_std)
        if patch_counts is None:
            patch_counts = torch.full((B,), N, dtype=torch.long, device=patches.device)
        patch_global, patch_counts = self._apply_patch_dropout(
            patch_global, patch_counts, self.config.patch_dropout_rate)
        slide_feat = F.normalize(self.scale_aggregators[scale_idx](
            patch_global, patch_counts=patch_counts, text_features=text_features,
            uot_module=self.uot_intra, concept_weights=concept_weights), dim=-1)

        raw_spatial_reshaped = slide_spatial = None
        if return_spatial and raw_spatial is not None:
            P_tok, hdim          = raw_spatial.shape[1], raw_spatial.shape[2]
            raw_spatial_reshaped = raw_spatial.view(B, N, P_tok, hdim)
            local_norm           = F.normalize(raw_spatial_reshaped, dim=-1)
            tile_mask            = (torch.arange(N, device=patches.device).unsqueeze(0)
                                    < patch_counts.unsqueeze(1))
            denom                = patch_counts.float().unsqueeze(-1).unsqueeze(-1).clamp(min=1)
            slide_spatial        = F.normalize(
                (local_norm * tile_mask.unsqueeze(-1).unsqueeze(-1).float()).sum(dim=1) / denom,
                dim=-1)
            del raw_spatial
        return slide_feat, patch_global, slide_spatial, raw_spatial_reshaped, patch_counts

    def forward_wsi(self, multi_patches, labels=None, multi_patch_counts=None):
        if not isinstance(multi_patches, dict):
            sn = self.scale_names[0]
            multi_patches       = {sn: multi_patches}
            multi_patch_counts  = {sn: multi_patch_counts}
        if multi_patch_counts is None:
            multi_patch_counts = {}

        B      = list(multi_patches.values())[0].shape[0]
        device = list(multi_patches.values())[0].device

        text_features   = self.nested_prompt_learner(level='both')
        concept_weights = text_features.get('concept_weights', None)

        swr_text_features = None
        if 'patch_features' in text_features and self._use_pl:
            tissue_stacked    = torch.cat(text_features['patch_features'], dim=0)
            swr_text_features = tissue_stacked.unsqueeze(0).expand(B, -1, -1)

        scale_slide_features   = []
        memory_spatial         = None
        memory_patch_feat      = None
        memory_patch_counts    = None
        memory_pertile_spatial = None

        for s_idx, sn in enumerate(self.scale_names):
            if sn not in multi_patches:
                continue
            is_mem     = (s_idx == self.memory_scale_idx)
            scale_text = (swr_text_features if self._swr_scale_flags[s_idx] else None)

            (slide_feat, patch_feat, slide_spatial,
             pertile_spatial, updated_counts) = self._encode_scale(
                multi_patches[sn], multi_patch_counts.get(sn, None), s_idx,
                return_spatial=is_mem,
                text_features=scale_text,
                concept_weights=concept_weights)

            scale_slide_features.append(slide_feat)

            if is_mem:
                memory_spatial         = slide_spatial
                memory_patch_feat      = patch_feat
                memory_patch_counts    = updated_counts
                memory_pertile_spatial = pertile_spatial
            else:
                del patch_feat, slide_spatial, pertile_spatial

        fused_slide = F.normalize(self.scale_fusion(scale_slide_features), dim=-1)

        d_patch_class = None
        if (self._use_pl and 'patch_features' in text_features
                and memory_patch_feat is not None and hasattr(self, 'tissue_mapper')):
            concept_stacked = torch.cat(text_features['patch_features'], dim=0)
            sim  = torch.einsum('bnd,td->bnt',
                                F.normalize(memory_patch_feat, dim=-1),
                                F.normalize(concept_stacked, dim=-1))
            _, N_s, _ = sim.shape
            if memory_patch_counts is not None:
                mask = (torch.arange(N_s, device=sim.device).unsqueeze(0)
                        < memory_patch_counts.unsqueeze(1)).unsqueeze(-1).expand_as(sim)
                ca   = (sim * mask.float()).sum(dim=1) / mask.sum(dim=1).float().clamp(min=1.0)
            else:
                ca = sim.mean(dim=1)
            d_patch_class = self.tissue_mapper(ca)
            del sim, ca

        d_direct = None
        if hasattr(self, 'direct_patch_class') and memory_patch_feat is not None:
            d_direct = self.direct_patch_class(memory_patch_feat, memory_patch_counts)

        del memory_patch_feat

        d_slide = None
        if self._use_sl and 'slide_features' in text_features:
            class_stacked = torch.cat(text_features['slide_features'], dim=0)
            d_slide       = torch.clamp(
                1 - torch.matmul(F.normalize(fused_slide, dim=-1),
                                 F.normalize(class_stacked, dim=-1).T), 0.0, 2.0)

        if (not self.config.disable_memory and self.memory is not None
                and self.memory.memory_initialized):
            if memory_spatial is not None:
                all_d = []
                for k in range(self.config.num_classes):
                    proto_dists = []
                    for p in range(self.memory.num_prototypes):
                        bary_kp = self.memory.barycenters[k, p]
                        if bary_kp.norm() < 1e-6:
                            proto_dists.append(torch.ones(B, device=device))
                            continue
                        bary_batch = bary_kp.unsqueeze(0).expand(B, -1, -1)
                        try:
                            d = self.uot_intra(memory_spatial, bary_batch)
                        except Exception:
                            d = 1 - torch.bmm(
                                F.normalize(memory_spatial, dim=-1, eps=1e-8),
                                F.normalize(bary_batch, dim=-1, eps=1e-8).transpose(1, 2)
                            ).mean(dim=[1, 2])
                        proto_dists.append(torch.clamp(d, 0.0, 2.0))
                    proto_stack = torch.stack(proto_dists, dim=1)
                    wk          = self.memory.prototype_weights[k].unsqueeze(0).to(device)
                    all_d.append((proto_stack * wk).sum(dim=1))
                d_I = torch.stack(all_d, dim=1)
            else:
                d_I = torch.ones(B, self.config.num_classes, device=device)

            if (self.training and self.config.memory_update_during_train
                    and not self.config.disable_ema and labels is not None
                    and memory_spatial is not None):
                self.memory.ema_update(memory_spatial, labels)
        else:
            d_I = torch.ones(B, self.config.num_classes, device=device)

        del memory_spatial, memory_pertile_spatial

        if d_slide is not None:
            mu         = torch.sigmoid(self.mu_slide)
            d_combined = mu * d_slide + (1 - mu) * d_I
        else:
            d_combined = d_I

        if d_patch_class is not None:
            av      = torch.sigmoid(self.alpha)
            d_final = av * d_combined + (1 - av) * d_patch_class
        else:
            d_final = d_combined

        if d_direct is not None:
            dw      = torch.sigmoid(self.direct_weight)
            d_final = (1 - dw) * d_final + dw * d_direct

        probs = self.compute_probability_from_distance(d_final)

        if labels is not None:
            smooth = self.config.label_smoothing
            n_cls  = self.config.num_classes
            st     = torch.full_like(probs, smooth / n_cls)
            st.scatter_(1, labels.unsqueeze(1), 1.0 - smooth + smooth / n_cls)
            loss_ce = -(st * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            cp      = -0.1 * entropy
            bi      = torch.arange(B, device=device)

            def margin_loss(d, labs):
                if d is None:
                    return torch.tensor(0.0, device=device)
                dp = d[bi, labs]
                m  = torch.ones_like(d, dtype=torch.bool)
                m[bi, labs] = False
                return F.relu(dp.unsqueeze(1) - d[m].view(B, -1) + 0.5).mean()

            lm_s = margin_loss(d_slide, labels)
            lm_I = margin_loss(d_I, labels)
            lm_p = margin_loss(d_patch_class, labels)
            lm_d = (margin_loss(d_direct, labels) if d_direct is not None
                    else torch.tensor(0.0, device=device))

            pdl = torch.tensor(0.0, device=device)
            if self.memory is not None and self.config.prototype_diversity_weight > 0:
                pdl = (self.memory.compute_prototype_diversity_loss()
                       * self.config.prototype_diversity_weight)

            if self.loss_weighter is not None:
                total_loss, lw = self.loss_weighter([loss_ce, lm_s, lm_I, lm_p, lm_d])
                total_loss     = total_loss + cp + pdl
            else:
                total_loss = (loss_ce + 0.1 * lm_s + 0.1 * lm_I
                              + 0.05 * lm_p + 0.05 * lm_d + cp + pdl)
                lw         = [1.0, 0.1, 0.1, 0.05, 0.05]

            return {"total_loss": total_loss, "loss_ce": loss_ce,
                    "loss_margin_slide": lm_s, "loss_margin_I": lm_I,
                    "loss_margin_patch": lm_p, "loss_margin_direct": lm_d,
                    "loss_proto_diversity": pdl, "confidence_penalty": cp,
                    "entropy": entropy, "logits": torch.log(probs + 1e-10),
                    "probs": probs, "loss_weights": lw, "d_slide": d_slide,
                    "d_I": d_I, "d_patch": d_patch_class, "d_direct": d_direct,
                    "d_final": d_final}

        return {"logits": torch.log(probs + 1e-10), "probs": probs,
                "d_slide": d_slide, "d_I": d_I, "d_patch": d_patch_class,
                "d_direct": d_direct, "d_final": d_final}


# ===========================================================================
# DATA – Indexing & Datasets
# ===========================================================================
class TCGAIDParser:
    TCGA_BARCODE = re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2,3})*(?:-DX\d+)?)')

    @staticmethod
    def parse(slide_id):
        raw = str(slide_id).strip()
        for ext in ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs']:
            if raw.lower().endswith(ext):
                raw = raw[:-len(ext)]
        parts = raw.split('.')
        if len(parts) > 1 and len(parts[-1]) > 20:
            raw = parts[0]
        m  = TCGAIDParser.TCGA_BARCODE.search(raw)
        bc = m.group(1) if m else raw
        c  = bc.split('-')
        return {'raw': raw, 'barcode': bc,
                'patient_id': '-'.join(c[:3]) if len(c) >= 3 else bc}


VALID_EXT = {'.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def _list_images_in(folder):
    paths = []
    try:
        for f in sorted(Path(folder).iterdir()):
            if f.is_file() and f.suffix.lower() in VALID_EXT:
                paths.append(str(f))
    except Exception:
        pass
    return paths


def list_pt_files_in_directory(directory):
    paths = []
    try:
        for f in sorted(Path(directory).iterdir()):
            if f.is_file() and f.suffix.lower() == '.pt':
                paths.append(str(f))
    except Exception:
        pass
    return paths


def index_rcc_dataset(rcc_root, class_dir_names, class_to_idx,
                      scales=('5x', '20x'), min_patches_per_scale=None,
                      conch_feats_root=None):
    if min_patches_per_scale is None:
        min_patches_per_scale = {s: 10 for s in scales}
    if conch_feats_root is None:
        conch_feats_root = "/home/datasets/tcga_rcc/conch_feats"

    indexed = []
    stats   = defaultdict(int)

    for cls_name, cls_dir in class_dir_names.items():
        label    = class_to_idx[cls_name]
        cls_root = Path(rcc_root) / cls_dir / "pyramid" / cls_dir
        if not cls_root.exists():
            logger.warning(f"Class root missing: {cls_root}")
            continue
        slide_dirs = sorted([d for d in cls_root.iterdir() if d.is_dir()])
        logger.info(f"Class {cls_name}: scanning {len(slide_dirs)} slides at {cls_root}")

        for slide_dir in tqdm(slide_dirs, desc=f"  {cls_name}"):
            scale_paths = {}
            ok          = True

            for s in scales:
                if s == '5x':
                    s_dir = slide_dir / "5x"
                    if not s_dir.exists():
                        ok = False; break
                    paths = _list_images_in(s_dir)
                    if len(paths) < min_patches_per_scale.get(s, 10):
                        ok = False; break
                    scale_paths[s] = paths
                    stats[f'{cls_name}_img_{s}'] += 1

                elif s == '20x':
                    feat_dir = (Path(conch_feats_root) / cls_dir / "pyramid"
                                / cls_dir / slide_dir.name / "20x")
                    if feat_dir.exists():
                        pt_files = list_pt_files_in_directory(feat_dir)
                        if len(pt_files) >= min_patches_per_scale.get(s, 30):
                            scale_paths[s] = pt_files
                            stats[f'{cls_name}_feat_{s}'] += 1
                        else:
                            stats[f'{cls_name}_feat_{s}_too_few'] += 1
                            ok = False; break
                    else:
                        single_pt = Path(conch_feats_root) / f"{slide_dir.name}.pt"
                        if single_pt.exists():
                            scale_paths[s] = str(single_pt)
                            stats[f'{cls_name}_feat_{s}_single'] += 1
                        else:
                            stats[f'{cls_name}_missing_20x_feat'] += 1
                            ok = False; break
                else:
                    continue

            if not ok or not scale_paths:
                stats[f'{cls_name}_skipped'] += 1
                continue

            sid   = slide_dir.name
            pid   = TCGAIDParser.parse(sid)['patient_id']
            total = len(scale_paths.get('5x', []))
            indexed.append({
                'slide_id':        sid,
                'label':           int(label),
                'class_name':      cls_name,
                'scale_paths':     scale_paths,
                'num_patches':     total,
                'available_scales': list(scale_paths.keys()),
                'patient_id':      pid,
            })

    logger.info(f"Indexing summary: {dict(stats)}")
    return indexed


# ---------- Split helpers ----------
def make_patient_level_splits(indexed_data, train_ratio=0.7, val_ratio=0.15,
                              test_ratio=0.15, seed=109, num_classes=3):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    patient_to_slides = defaultdict(list)
    for item in indexed_data:
        patient_to_slides[item['patient_id']].append(item)

    patients_by_class = defaultdict(list)
    for pid, slides in patient_to_slides.items():
        labs     = [s['label'] for s in slides]
        majority = max(set(labs), key=labs.count)
        patients_by_class[majority].append(pid)

    train_slides, val_slides, test_slides = [], [], []
    for cls in range(num_classes):
        pids    = list(patients_by_class.get(cls, []))
        rng.shuffle(pids)
        n       = len(pids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        for pid in pids[:n_train]:
            train_slides.extend(patient_to_slides[pid])
        for pid in pids[n_train:n_train + n_val]:
            val_slides.extend(patient_to_slides[pid])
        for pid in pids[n_train + n_val:]:
            test_slides.extend(patient_to_slides[pid])

    rng.shuffle(train_slides); rng.shuffle(val_slides); rng.shuffle(test_slides)
    return train_slides, val_slides, test_slides


def save_splits_to_json(train_data, val_data, test_data, json_path):
    def slim(items):
        return [{'slide_id': it['slide_id'], 'label': it['label'],
                 'class_name': it['class_name'], 'patient_id': it['patient_id']}
                for it in items]
    with open(json_path, 'w') as f:
        json.dump({'train': slim(train_data), 'val': slim(val_data),
                   'test': slim(test_data)}, f, indent=2)
    logger.info(f"Saved splits → {json_path}")


def load_splits_from_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d['train'], d['val'], d['test']


def attach_paths_to_split(slim_split, all_indexed):
    by_id = {it['slide_id']: it for it in all_indexed}
    out   = []
    for item in slim_split:
        sid = item['slide_id']
        if sid in by_id:
            out.append(by_id[sid])
        else:
            logger.warning(f"Slide {sid} from saved split not found on disk; skipping.")
    return out


# ===========================================================================
# MultiScaleWSIDataset
# ===========================================================================
class MultiScaleWSIDataset(Dataset):
    def __init__(self, indexed_data, transform=None, is_train=True, config=None):
        self.transform  = transform
        self.is_train   = is_train
        self.config     = config
        self.scales     = config.scales
        self.scale_max  = config.scale_max_patches
        self.scale_min  = config.scale_min_patches
        self.img_size   = int(getattr(config, 'conch_img_size', 224))
        self.data = [it for it in indexed_data
                     if all(it['scale_paths'].get(s) is not None for s in self.scales)]
        logger.info(f"MultiScaleWSIDataset: {len(self.data)} slides (img_size={self.img_size})")
        self.epoch_samples = []
        self.reshuffle()

    def reshuffle(self):
        self.epoch_samples = []
        for it in self.data:
            sample = {'slide_id': it['slide_id'], 'label': it['label'], 'scale_paths': {}}
            for s in self.scales:
                val = it['scale_paths'].get(s)
                if s == '20x' and isinstance(val, list) and all(p.endswith('.pt') for p in val):
                    mx = self.scale_max.get(s, 3000)
                    if len(val) > mx and self.is_train:
                        sel = random.sample(val, mx)
                    elif len(val) > mx:
                        sel = [val[i] for i in np.linspace(0, len(val)-1, mx, dtype=int)]
                    else:
                        sel = list(val)
                    if self.is_train:
                        random.shuffle(sel)
                    sample['scale_paths'][s] = sel
                elif isinstance(val, str) and val.endswith('.pt'):
                    sample['scale_paths'][s] = val
                else:
                    paths = val if isinstance(val, list) else []
                    mx    = self.scale_max.get(s, 800)
                    if len(paths) > mx and self.is_train:
                        sel = random.sample(paths, mx)
                    elif len(paths) > mx:
                        sel = [paths[i] for i in np.linspace(0, len(paths)-1, mx, dtype=int)]
                    else:
                        sel = list(paths)
                    if self.is_train:
                        random.shuffle(sel)
                    sample['scale_paths'][s] = sel
            self.epoch_samples.append(sample)
        if self.is_train:
            random.shuffle(self.epoch_samples)

    def __len__(self):
        return len(self.epoch_samples)

    def _load(self, scale_item):
        if isinstance(scale_item, str) and scale_item.endswith('.pt'):
            feat = torch.load(scale_item, map_location='cpu')
            if feat.dim() == 2:
                return feat
            elif feat.dim() == 3 and feat.shape[0] == 1 and feat.shape[2] == self.config.embedding_dim:
                return feat.squeeze(0)
            raise RuntimeError(f"Unexpected feature tensor shape {feat.shape} from {scale_item}")
        elif isinstance(scale_item, list) and all(p.endswith('.pt') for p in scale_item):
            features = []
            for p in scale_item:
                feat = torch.load(p, map_location='cpu')
                if feat.dim() == 1:
                    features.append(feat)
                elif feat.dim() == 2 and feat.shape[0] == 1:
                    features.append(feat.squeeze(0))
                else:
                    raise RuntimeError(f"Unexpected per-patch feature shape {feat.shape} from {p}")
            if not features:
                return torch.zeros(0, self.config.embedding_dim)
            return torch.stack(features, dim=0)
        else:
            patches = []
            tgt     = (self.img_size, self.img_size)
            for p in scale_item:
                try:
                    with Image.open(p) as img:
                        img = img.convert('RGB')
                        if img.size != tgt:
                            img = img.resize(tgt, Image.Resampling.BILINEAR)
                        patches.append(self.transform(img) if self.transform
                                       else transforms.ToTensor()(img))
                except Exception:
                    continue
            if not patches:
                return torch.zeros(1, 3, self.img_size, self.img_size)
            return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        s = self.epoch_samples[idx]
        return ({sc: self._load(s['scale_paths'][sc]) for sc in self.scales},
                s['label'], s['slide_id'])


def collate_multiscale_batch(batch):
    scales = list(batch[0][0].keys())
    B      = len(batch)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    sids   = [b[2] for b in batch]
    mp, mc = {}, {}
    for s in scales:
        first = batch[0][0][s]
        if isinstance(first, torch.Tensor) and first.dim() == 2:
            ts = [b[0][s] for b in batch]
            cs = torch.tensor([t.shape[0] for t in ts], dtype=torch.long)
            mn = cs.max().item()
            D  = ts[0].shape[1]
            pad = torch.zeros(B, mn, D, dtype=ts[0].dtype)
            for i, t in enumerate(ts):
                pad[i, :t.shape[0]] = t
            mp[s] = pad
            mc[s] = cs
        else:
            ts = [b[0][s] for b in batch]
            cs = torch.tensor([t.shape[0] for t in ts], dtype=torch.long)
            mn = cs.max().item()
            C_c, H, W = ts[0].shape[1], ts[0].shape[2], ts[0].shape[3]
            pad = torch.zeros(B, mn, C_c, H, W, dtype=ts[0].dtype)
            for i, t in enumerate(ts):
                pad[i, :t.shape[0]] = t
            mp[s] = pad
            mc[s] = cs
    return mp, labels, sids, mc


class FilteredMemoryDataset(Dataset):
    def __init__(self, filtered_data, memory_scale, transform=None,
                 max_per_slide=3, img_size=224):
        self.transform = transform
        self.img_size  = int(img_size)
        self.samples   = []
        for item in filtered_data:
            val = item['scale_paths'].get(memory_scale)
            if val is None:
                continue
            if isinstance(val, list) and all(
                    isinstance(p, str) and not p.endswith('.pt') for p in val):
                for p in val[:min(max_per_slide, len(val))]:
                    self.samples.append((p, item['label']))
        logger.info(f"FilteredMemoryDataset: {len(self.samples)} tiles from "
                    f"{len(filtered_data)} slides (scale: {memory_scale}, "
                    f"img_size: {self.img_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tgt         = (self.img_size, self.img_size)
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if img.size != tgt:
                    img = img.resize(tgt, Image.Resampling.BILINEAR)
                return (self.transform(img) if self.transform
                        else transforms.ToTensor()(img)), label
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size), label


# ===========================================================================
# Balanced Sampler & Few-shot subset helpers
# ===========================================================================
def create_balanced_sampler(indexed_data, strategy='class_and_sqrt_size'):
    labels = [it['label']       for it in indexed_data]
    counts = [it['num_patches'] for it in indexed_data]
    cf     = defaultdict(int)
    for l in labels:
        cf[l] += 1
    cw      = {c: 1.0 / n for c, n in cf.items()}
    weights = []
    for l, c in zip(labels, counts):
        w = cw[l]
        if strategy == 'class_and_sqrt_size':
            w *= 1.0 / max(np.sqrt(c), 1)
        elif strategy == 'class_and_inverse':
            w *= 1.0 / max(c, 1)
        weights.append(w)
    t       = sum(weights)
    weights = [w / t for w in weights]
    return WeightedRandomSampler(weights, num_samples=len(indexed_data), replacement=True)


def create_few_shot_subset(indexed_data, k_per_class):
    ci = defaultdict(list)
    for i, it in enumerate(indexed_data):
        ci[it['label']].append(i)
    sel = []
    for l, ids in ci.items():
        sel.extend(random.sample(ids, k_per_class) if len(ids) >= k_per_class else ids)
    return [indexed_data[i] for i in sel]


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_wsi(model, val_loader, config):
    model.eval()
    all_p, all_l, cor, tot = [], [], 0, 0
    total_loss, num_batches = 0.0, 0
    all_preds = []
    with torch.no_grad():
        for bd in tqdm(val_loader, desc="Validation"):
            mp, lb, _, mc = bd
            mp = {s: p.to(config.device) for s, p in mp.items()}
            mc = {s: c.to(config.device) for s, c in mc.items()}
            lb = lb.to(config.device)
            out  = model.forward_wsi(mp, labels=lb, multi_patch_counts=mc)
            pred = torch.argmax(out['probs'], dim=1)
            cor  += (pred == lb).sum().item()
            tot  += lb.size(0)
            all_p.append(out['probs'].cpu().numpy())
            all_l.extend(lb.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            total_loss  += out['total_loss'].item()
            num_batches += 1
    acc          = cor / tot if tot > 0 else 0
    bal_acc      = balanced_accuracy_score(all_l, all_preds) if len(all_l) > 0 else 0.0
    all_probs_arr = np.concatenate(all_p, axis=0) if all_p else np.zeros((0, config.num_classes))
    try:
        auc = roc_auc_score(all_l, all_probs_arr, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    try:
        f1 = f1_score(all_l, all_preds, average='macro')
    except Exception:
        f1 = 0.0
    return {'acc': acc, 'auc': auc, 'f1': f1,
            'val_loss':     total_loss / max(num_batches, 1),
            'balanced_acc': bal_acc,
            'all_probs':    all_probs_arr.tolist(),
            'all_labels':   all_l,
            'all_preds':    all_preds}


def evaluate_test_full_report(model, test_loader, config, output_dir="./test_results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    class_names  = config.class_names
    all_probs, all_labels, all_preds, all_slide_ids = [], [], [], []
    with torch.no_grad():
        for bd in tqdm(test_loader, desc="Testing"):
            mp, lb, sids, mc = bd
            mp   = {s: p.to(config.device) for s, p in mp.items()}
            mc   = {s: c.to(config.device) for s, c in mc.items()}
            lb   = lb.to(config.device)
            out  = model.forward_wsi(mp, labels=None, multi_patch_counts=mc)
            preds = torch.argmax(out["probs"], dim=1)
            all_probs.extend(out["probs"].cpu().numpy())
            all_labels.extend(lb.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_slide_ids.extend(sids)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_w    = f1_score(all_labels, all_preds, average="weighted")
    f1_m    = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(
            np.eye(config.num_classes)[all_labels], all_probs, average='macro')
    except Exception:
        ap = 0.0
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"\n{'='*60}\nTEST REPORT\n{'='*60}\n{report}")
    print(f"Accuracy: {acc:.4f} | Bal Acc: {bal_acc:.4f} | AUC(ovr macro): {auc:.4f} | "
          f"AP: {ap:.4f} | F1(macro): {f1_m:.4f} | F1(weighted): {f1_w:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    df_dict = {
        "slide_id":   all_slide_ids,
        "true_label": all_labels,
        "pred_label": all_preds,
        "true_class": [class_names[l] for l in all_labels],
        "pred_class": [class_names[p] for p in all_preds],
        "correct":    (all_labels == all_preds).astype(int),
    }
    for ci, cn in enumerate(class_names):
        df_dict[f"prob_{cn}"] = all_probs[:, ci]
    pd.DataFrame(df_dict).to_csv(
        os.path.join(output_dir, "test_predictions.csv"), index=False)
    summary = {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "auc_ovr_macro": float(auc), "average_precision_macro": float(ap),
        "f1_macro": float(f1_m), "f1_weighted": float(f1_w),
        "confusion_matrix": cm.tolist(), "class_names": class_names,
        "classification_report": report,
    }
    with open(os.path.join(output_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ===========================================================================
# Checkpointing
# ===========================================================================
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_acc, best_bal_acc, best_val_loss, patience_counter,
                    history, swa_model=None):
    state = {
        'epoch':             epoch,
        'model_state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc':          best_acc,
        'best_bal_acc':      best_bal_acc,
        'best_val_loss':     best_val_loss,
        'patience_counter':  patience_counter,
        'history':           dict(history),
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if swa_model is not None:
        state['swa_model_state'] = swa_model.model_state
        state['swa_n_averaged']  = swa_model.n_averaged
    rng = {'torch': torch.get_rng_state(),
           'numpy': np.random.get_state(),
           'python': random.getstate()}
    if torch.cuda.is_available():
        rng['cuda'] = torch.cuda.get_rng_state_all()
    state['rng_state'] = rng
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    swa_model=None, device='cpu'):
    """Load checkpoint and correctly restore BarycentreMemory state.

    v3.1 addition: after load_state_dict (strict=False) we call
    restore_initialized_flag_from_buffers() so that both new-format
    checkpoints (that carry _initialized_flag) and old-format checkpoints
    (that don't, but have populated barycenters) resume correctly.
    """
    logger.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # strict=False: tolerate new buffers absent in old checkpoints and vice-versa
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # ---- v3.1 RESUME FIX: restore memory_initialized from buffer or heuristic ----
    if hasattr(model, 'memory') and model.memory is not None:
        try:
            model.memory.restore_initialized_flag_from_buffers()
        except Exception as e:
            logger.warning(f"restore_initialized_flag_from_buffers failed: {e}")

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if swa_model is not None and 'swa_model_state' in ckpt:
        swa_model.model_state = ckpt['swa_model_state']
        swa_model.n_averaged  = ckpt.get('swa_n_averaged', 0)
    if 'rng_state' in ckpt:
        try:
            rng = ckpt['rng_state']
            torch.set_rng_state(rng['torch'])
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all(rng['cuda'])
        except Exception as e:
            logger.warning(f"Failed to restore RNG state: {e}")
    return {
        'epoch':            ckpt.get('epoch', 0),
        'best_acc':         ckpt.get('best_acc', 0.0),
        'best_bal_acc':     ckpt.get('best_bal_acc', 0.0),
        'best_val_loss':    ckpt.get('best_val_loss', float('inf')),
        'patience_counter': ckpt.get('patience_counter', 0),
        'history':          defaultdict(list, ckpt.get('history', {})),
    }


# ===========================================================================
# Training
# ===========================================================================
def train_one_epoch(model, train_loader, optimizer, scaler, config, epoch):
    model.train()
    model.set_epoch(epoch)
    train_loader.dataset.reshuffle()
    accum  = config.gradient_accumulation_steps
    el, ec, tcor, ttot, nb = 0., 0., 0, 0, 0
    loss_details = defaultdict(float)
    optimizer.zero_grad()
    num_passes = getattr(config, 'train_passes_per_epoch', 1)

    for pass_idx in range(num_passes):
        if pass_idx > 0:
            train_loader.dataset.reshuffle()
        all_batches = list(train_loader)
        pbar        = tqdm(all_batches, desc=f"Train E{epoch+1} P{pass_idx+1}/{num_passes}")
        for bi, bd in enumerate(pbar):
            global_bi    = pass_idx * len(all_batches) + bi
            mp, lb, _, mc = bd
            mp = {s: p.to(config.device) for s, p in mp.items()}
            mc = {s: c.to(config.device) for s, c in mc.items()}
            lb = lb.to(config.device)

            if config.mixed_precision and scaler:
                with autocast('cuda'):
                    out  = model.forward_wsi(mp, labels=lb, multi_patch_counts=mc)
                    loss = out['total_loss'] / accum
                    if getattr(config, 'rdrop_alpha', 0) > 0 and model.training:
                        out2 = model.forward_wsi(mp, labels=lb, multi_patch_counts=mc)
                        p1   = out['probs'].clamp(1e-8, 1.0)
                        p2   = out2['probs'].clamp(1e-8, 1.0)
                        kl1  = F.kl_div(p1.log(), p2, reduction='batchmean')
                        kl2  = F.kl_div(p2.log(), p1, reduction='batchmean')
                        loss = loss + config.rdrop_alpha * (kl1 + kl2) / (2 * accum)
                scaler.scale(loss).backward()
                if (global_bi + 1) % accum == 0 or (bi + 1) == len(all_batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                out  = model.forward_wsi(mp, labels=lb, multi_patch_counts=mc)
                loss = out['total_loss'] / accum
                loss.backward()
                if (global_bi + 1) % accum == 0 or (bi + 1) == len(all_batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            pred   = torch.argmax(out['probs'], dim=1)
            tcor  += (pred == lb).sum().item()
            ttot  += lb.size(0)
            el    += out['total_loss'].item()
            ec    += out['loss_ce'].item()
            nb    += 1
            for k in ['loss_margin_slide', 'loss_margin_I', 'loss_margin_patch',
                      'loss_margin_direct', 'loss_proto_diversity']:
                if k in out:
                    loss_details[k] += out[k].item()
            pbar.set_postfix(loss=f"{out['total_loss'].item():.4f}",
                             acc=f"{tcor/max(ttot,1):.3f}")

    el /= max(nb, 1)
    ec /= max(nb, 1)
    ta  = tcor / max(ttot, 1)
    for k in loss_details:
        loss_details[k] /= max(nb, 1)

    extras = {}
    for attr in ['tau', 'mu_slide', 'alpha']:
        if hasattr(model, attr):
            v = getattr(model, attr)
            extras[attr] = (torch.sigmoid(v).item() if attr != 'tau' else v.item())

    if hasattr(model, 'memory') and model.memory is not None:
        extras['ema_momentum'] = model.memory.get_momentum()
        extras['ema_updates']  = model.memory._update_count
        diag = model.memory.get_diagnostics()
        for c in range(model.memory.num_classes):
            extras[f'c{c}_avg_update_mag'] = diag.get(f'c{c}_avg_update_mag', 0.0)
            extras[f'c{c}_avg_drift']      = diag.get(f'c{c}_avg_drift', 0.0)
            extras[f'c{c}_inter_proto_sim'] = diag.get(f'c{c}_inter_proto_sim', 0.0)
            extras[f'c{c}_buf_size']        = diag.get(f'c{c}_buf_size', 0)

    return {'loss': el, 'loss_ce': ec, 'train_acc': ta,
            'lr': optimizer.param_groups[0]['lr'],
            **{k: v for k, v in loss_details.items()}, **extras}


# ===========================================================================
# MAIN
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', action='store_true',
                   help='Resume from last_checkpoint.pth in save_dir.')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Explicit checkpoint path to resume from.')
    p.add_argument('--rebuild_splits', action='store_true',
                   help='Rebuild train/val/test split JSON.')
    p.add_argument('--k', type=int, default=None,
                   help='Override few_shot_k_train.')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override num_epochs.')
    return p.parse_args()


def main(cfg=None, args=None):
    if cfg is None:
        cfg = Config()
    if args is None:
        args = parse_args()

    if args.k is not None:
        cfg.few_shot_k_train = args.k
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    cfg.resume        = args.resume
    cfg.resume_from   = args.resume_from
    cfg.rebuild_splits = args.rebuild_splits

    cfg = apply_shot_adaptive_config(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("MULTI-SCALE MUOT-CONCH — TCGA-RCC 3-class (KICH/KIRP/KIRC) v3.1")
    print(f"  k={cfg.few_shot_k_train} | GPU: cuda:{cfg.gpu_device_id}")
    print(f"  Routing concepts  : {len(build_routing_concepts())} (RCC-specific)")
    print(f"  Pre-extracted 20× features from per-patch .pt files")
    print(f"  Memory scale      : {cfg.memory_scale}")
    print(f"  Save dir          : {cfg.save_dir}")
    print("=" * 80 + "\n")

    print(f"\n[Indexing] Walking {cfg.rcc_root} ...")
    all_indexed = index_rcc_dataset(
        cfg.rcc_root, cfg.class_dir_names, cfg.class_to_idx,
        scales=tuple(cfg.scales), min_patches_per_scale=cfg.scale_min_patches,
        conch_feats_root=cfg.conch_feats_root)
    if not all_indexed:
        logger.error("No slides indexed — check RCC paths.")
        return None

    n_per_class = defaultdict(int)
    for it in all_indexed:
        n_per_class[it['class_name']] += 1
    print(f"  Slides per class indexed: {dict(n_per_class)}")
    print(f"  Total: {len(all_indexed)} slides")

    if cfg.rebuild_splits or not os.path.exists(cfg.split_json):
        print("\n[Splits] Generating new patient-level stratified splits...")
        train_full, val_full, test_full = make_patient_level_splits(
            all_indexed, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
            seed=cfg.split_seed, num_classes=cfg.num_classes)
        save_splits_to_json(train_full, val_full, test_full, cfg.split_json)
    else:
        print(f"\n[Splits] Loading existing splits from {cfg.split_json}")
        train_slim, val_slim, test_slim = load_splits_from_json(cfg.split_json)
        train_full = attach_paths_to_split(train_slim, all_indexed)
        val_full   = attach_paths_to_split(val_slim,   all_indexed)
        test_full  = attach_paths_to_split(test_slim,  all_indexed)

    for name, split in [('train', train_full), ('val', val_full), ('test', test_full)]:
        per_cls = defaultdict(int)
        for it in split:
            per_cls[it['class_name']] += 1
        print(f"  {name:5s}: {len(split):4d} slides  per-class={dict(per_cls)}")

    train_data = (create_few_shot_subset(train_full, cfg.few_shot_k_train)
                  if cfg.few_shot_train else train_full)
    fs_counts  = defaultdict(int)
    for item in train_data:
        fs_counts[cfg.idx_to_class[item['label']]] += 1
    logger.info(f"Final training set (k={cfg.few_shot_k_train}): {dict(fs_counts)}")

    MEAN = (0.48145466, 0.4578275,  0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)
    train_tl = [transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(90)]
    if cfg.use_stain_augmentation:
        train_tl.append(StainAugmentation(0.05, 0.1))
    train_tl.extend([transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                     transforms.RandomGrayscale(p=0.05),
                     transforms.ToTensor(),
                     transforms.Normalize(MEAN, STD)])
    tt = transforms.Compose(train_tl)
    vt = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_ds = MultiScaleWSIDataset(train_data, tt, True,  cfg)
    val_ds   = MultiScaleWSIDataset(val_full,   vt, False, cfg)
    test_ds  = MultiScaleWSIDataset(test_full,  vt, False, cfg)

    sampler = (create_balanced_sampler(train_ds.data, cfg.slide_sampler_strategy)
               if cfg.slide_sampler_strategy != 'none' else None)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                    shuffle=(sampler is None), num_workers=cfg.num_workers,
                    collate_fn=collate_multiscale_batch, pin_memory=cfg.pin_memory,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)
    vl = DataLoader(val_ds, batch_size=max(1, cfg.batch_size), shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=collate_multiscale_batch,
                    pin_memory=cfg.pin_memory, persistent_workers=cfg.num_workers > 0)
    test_dl = DataLoader(test_ds, batch_size=max(1, cfg.batch_size), shuffle=False,
                         num_workers=cfg.num_workers, collate_fn=collate_multiscale_batch,
                         pin_memory=cfg.pin_memory, persistent_workers=cfg.num_workers > 0)

    # ---------- Model ----------
    model = MultiScaleMUOT_CONCH(cfg).to(cfg.device)
    tp    = sum(p.numel() for p in model.parameters())
    tr    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {tp:,} total, {tr:,} trainable ({tr/tp*100:.2f}%)")
    print_gpu_status(cfg.device, "after model load")

    # ---------- Determine resume path ----------
    last_ckpt_path = os.path.join(cfg.save_dir, "last_checkpoint.pth")
    resume_path    = None
    if cfg.resume_from is not None and os.path.exists(cfg.resume_from):
        resume_path = cfg.resume_from
    elif cfg.resume and os.path.exists(last_ckpt_path):
        resume_path = last_ckpt_path
    do_resume = resume_path is not None

    # ---------- Optimizer / Scheduler / Scaler ----------
    gc_map = {
        'visual_encoder':                         ('vp', 0.5),
        'nested_prompt_learner.routing_soft_ctx': ('rc', 1.0),
        'nested_prompt_learner.slide_soft_ctx':   ('sc', 1.0),
        'scale_aggregators':                      ('ag', 2.0),
        'scale_fusion':                           ('fu', 2.0),
        'tissue_mapper':                          ('mp', 2.0),
    }
    grp = defaultdict(list)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        matched = False
        for k, (g, _) in gc_map.items():
            if k in n:
                grp[g].append(p); matched = True; break
        if not matched:
            grp['other'].append(p)

    glr = {'vp': 0.5, 'rc': 1.0, 'sc': 1.0, 'ag': 2.0, 'fu': 2.0, 'mp': 2.0, 'other': 0.1}
    pg  = [{'params': ps, 'lr': cfg.learning_rate * glr.get(g, 0.1)}
           for g, ps in grp.items()]
    opt = torch.optim.AdamW(pg, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(10, cfg.num_epochs // 4), T_mult=2, eta_min=cfg.min_lr)
    scaler    = GradScaler() if cfg.mixed_precision else None
    swa_model = SWAModel(model) if cfg.swa_enabled else None

    start_epoch     = 0
    best_val_loss   = float('inf')
    best_acc        = 0.0
    best_bal_acc    = 0.0
    patience_counter = 0
    history         = defaultdict(list)

    # ---------- Resume (load checkpoint first, then check memory) ----------
    if do_resume:
        # load_checkpoint already calls restore_initialized_flag_from_buffers()
        info          = load_checkpoint(resume_path, model, opt, scheduler, scaler,
                                        swa_model, device=cfg.device)
        start_epoch   = info['epoch'] + 1
        best_acc      = info['best_acc']
        best_bal_acc  = info['best_bal_acc']
        best_val_loss = info['best_val_loss']
        patience_counter = info['patience_counter']
        history       = info['history']
        print(f"\n>> Resumed from epoch {info['epoch']} (next: {start_epoch})  "
              f"best_acc={best_acc:.4f}  best_bal_acc={best_bal_acc:.4f}")

        # After loading, check if memory is properly initialized.
        # For new-format checkpoints: _initialized_flag was restored from state_dict.
        # For old-format checkpoints: restore_initialized_flag_from_buffers() handled it.
        # If still not initialized (extremely old checkpoint with zero barycenters),
        # rebuild from scratch.
        if model.memory is not None and not model.memory.memory_initialized:
            print("\n>> Memory not initialized after checkpoint load "
                  "→ rebuilding from training data ...")
            mem_ds = FilteredMemoryDataset(train_ds.data, cfg.memory_scale, vt,
                                           img_size=cfg.conch_img_size)
            mem_dl = DataLoader(mem_ds, batch_size=32, shuffle=False,
                                num_workers=2, pin_memory=True)
            model.init_memory(mem_dl)
        else:
            print(f">> Memory already initialized "
                  f"(updates so far: {model.memory._update_count if model.memory else 'N/A'})")
        print()

    else:
        # Fresh run — initialize memory from training tiles
        mem_ds = FilteredMemoryDataset(train_ds.data, cfg.memory_scale, vt,
                                       img_size=cfg.conch_img_size)
        mem_dl = DataLoader(mem_ds, batch_size=32, shuffle=False,
                            num_workers=2, pin_memory=True)
        model.init_memory(mem_dl)

    # ---------- Training loop ----------
    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{cfg.num_epochs}\n{'='*60}")
            train_metrics = train_one_epoch(model, tl, opt, scaler, cfg, epoch)
            scheduler.step()
            for k, v in train_metrics.items():
                history[k].append(v)

            overfit_flag = ""
            if train_metrics['train_acc'] > 0.95 and epoch > cfg.min_train_epochs // 2:
                overfit_flag = " ⚠️ OVERFIT"
            print(f"Epoch {epoch+1}: Loss={train_metrics['loss']:.4f}  "
                  f"CE={train_metrics['loss_ce']:.4f}  "
                  f"Acc={train_metrics['train_acc']:.4f}  "
                  f"LR={train_metrics['lr']:.6f}{overfit_flag}")

            if 'ema_momentum' in train_metrics:
                print(f"  Memory: momentum={train_metrics['ema_momentum']:.4f}  "
                      f"updates={train_metrics['ema_updates']}")
                for c in range(cfg.num_classes):
                    avg_mag   = train_metrics.get(f'c{c}_avg_update_mag', 0)
                    avg_drift = train_metrics.get(f'c{c}_avg_drift', 0)
                    inter_sim = train_metrics.get(f'c{c}_inter_proto_sim', 0)
                    buf_size  = train_metrics.get(f'c{c}_buf_size', 0)
                    if avg_mag < 0.001:     status = "⚠️ FROZEN"
                    elif inter_sim > 0.8:   status = "⚠️ COLLAPSED"
                    elif avg_drift > 0.5:   status = "⚠️ DRIFTED"
                    else:                   status = "✓"
                    print(f"    {cfg.idx_to_class[c]}: mag={avg_mag:.5f}  "
                          f"drift={avg_drift:.4f}  sim={inter_sim:.4f}  "
                          f"buf={buf_size}  {status}")

            if (cfg.swa_enabled and swa_model is not None
                    and epoch >= cfg.swa_start_epoch):
                swa_model.update(model)
                if swa_model.n_averaged % 5 == 0:
                    print(f"  SWA: averaged {swa_model.n_averaged} checkpoints")

            if (epoch + 1) % cfg.val_frequency == 0 or epoch == cfg.num_epochs - 1:
                val_results = evaluate_wsi(model, vl, cfg)
                history['val_acc'].append(val_results['acc'])
                history['val_bal_acc'].append(val_results['balanced_acc'])
                history['val_auc'].append(val_results['auc'])
                history['val_f1'].append(val_results['f1'])
                history['val_loss'].append(val_results['val_loss'])
                history['val_epoch'].append(epoch + 1)

                overfit_gap = train_metrics['train_acc'] - val_results['acc']
                history['overfit_gap'].append(overfit_gap)
                gap_warn = (f" ⚠️ GAP={overfit_gap:.3f}"
                            if overfit_gap > cfg.overfit_gap_threshold else "")
                print(f"  Val: Acc={val_results['acc']:.4f}  "
                      f"BalAcc={val_results['balanced_acc']:.4f}  "
                      f"AUC={val_results['auc']:.4f}  "
                      f"F1={val_results['f1']:.4f}  "
                      f"Loss={val_results['val_loss']:.4f}{gap_warn}")
                print_gpu_status(cfg.device, "val")

                if (cfg.swa_enabled and swa_model is not None
                        and swa_model.n_averaged > 0):
                    swa_eval = copy.deepcopy(model)
                    swa_model.apply(swa_eval)
                    swa_res = evaluate_wsi(swa_eval, vl, cfg)
                    print(f"  SWA: Acc={swa_res['acc']:.4f}  "
                          f"BalAcc={swa_res['balanced_acc']:.4f}  "
                          f"AUC={swa_res['auc']:.4f}")
                    if swa_res['balanced_acc'] > val_results['balanced_acc']:
                        val_results = swa_res
                        print("  >> SWA is better — using for saving")
                    del swa_eval

                if val_results['acc'] > best_acc:
                    best_acc = val_results['acc']
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'best_acc': best_acc, 'best_bal_acc': best_bal_acc,
                                'history': dict(history)},
                               os.path.join(cfg.save_dir, "best_acc_model.pth"))
                    print(f"  >> Best ACCURACY saved (Acc={best_acc:.4f})")

                if val_results['balanced_acc'] > best_bal_acc:
                    best_bal_acc = val_results['balanced_acc']
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'best_acc': best_acc, 'best_bal_acc': best_bal_acc,
                                'history': dict(history)},
                               os.path.join(cfg.save_dir, "best_bal_acc_model.pth"))
                    print(f"  >> Best BAL-ACC saved (BalAcc={best_bal_acc:.4f})")

                if val_results['val_loss'] < best_val_loss:
                    best_val_loss    = val_results['val_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch >= cfg.min_train_epochs and cfg.early_stopping_patience > 0
                        and patience_counter >= cfg.early_stopping_patience):
                    print(f"\n  >> EARLY STOPPING at epoch {epoch+1}")
                    save_checkpoint(last_ckpt_path, model, opt, scheduler, scaler,
                                    epoch, best_acc, best_bal_acc, best_val_loss,
                                    patience_counter, history, swa_model)
                    break

            save_checkpoint(last_ckpt_path, model, opt, scheduler, scaler,
                            epoch, best_acc, best_bal_acc, best_val_loss,
                            patience_counter, history, swa_model)

    except KeyboardInterrupt:
        print("\n>> Training interrupted — last_checkpoint.pth is up to date.")
    except Exception as e:
        print(f"\n>> Training failed: {e}")
        try:
            save_checkpoint(last_ckpt_path, model, opt, scheduler, scaler,
                            epoch, best_acc, best_bal_acc, best_val_loss,
                            patience_counter, history, swa_model)
            print(">> Emergency checkpoint saved.")
        except Exception:
            pass
        raise

    if cfg.swa_enabled and swa_model is not None and swa_model.n_averaged > 0:
        swa_final = copy.deepcopy(model)
        swa_model.apply(swa_final)
        torch.save({'model_state_dict': swa_final.state_dict(),
                    'n_averaged':  swa_model.n_averaged,
                    'best_acc':    best_acc,
                    'best_bal_acc': best_bal_acc},
                   os.path.join(cfg.save_dir, "swa_model.pth"))
        del swa_final

    with open(os.path.join(cfg.save_dir, "history.json"), 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v]
                   for k, v in history.items()}, f, indent=2)

    with open(os.path.join(cfg.save_dir, "config_used.json"), 'w') as f:
        json.dump({k: v for k, v in vars(cfg).items()
                   if not k.startswith('_')
                   and isinstance(v, (int, float, str, bool, list, dict, type(None)))},
                  f, indent=2)

    print(f"\nDone! Best Acc={best_acc:.4f}, Best BalAcc={best_bal_acc:.4f}")

    best_path = os.path.join(cfg.save_dir, "best_bal_acc_model.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(cfg.save_dir, "best_acc_model.pth")
    if os.path.exists(best_path) and test_full:
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded best model from epoch {ckpt.get('epoch', '?')+1}")
        evaluate_test_full_report(model, test_dl, cfg,
                                  os.path.join(cfg.save_dir, "test_results"))

    return {'best_acc': best_acc, 'best_bal_acc': best_bal_acc}


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(54)
    np.random.seed(54)
    random.seed(54)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(54)
        torch.cuda.manual_seed_all(54)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    main(args=args)