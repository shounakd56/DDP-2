"""
extract_pi.py
-------------
Shared low-level utilities for the interpretability scripts.

What this provides
------------------
- load_trained_model()   : load Config, build model, load .pth weights.
- get_concept_text_features(model)
                         : run the prompt learner once, return
                           (concept_names, text_features (C, D), prior_w (C,),
                            learned_imp (C,)) on CPU.
- encode_slide_at_scale(model, slide_item, scale, transform)
                         : run the visual encoder on the patches of ONE slide.
                           Returns (patch_features (N, D), patch_paths or None,
                           patch_counts).
- compute_pi(model, patch_features, text_features)
                         : returns the UOT plan π (N, C) for ONE slide.
                           Optionally returns the SWR-effective routing
                           (post concept_weights & concept_importance gating).
- pick_slides_for_class(split_data, num_per_class)
                         : helper to grab K slides per class from the
                           pre-built split JSON.

These functions are designed so a single small script can call them
without re-implementing model loading or transport extraction.
"""

from __future__ import annotations
import os
import sys
import json
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_trained_model(checkpoint_path: str,
                       main_module: str,
                       device: str = "cuda:0",
                       overrides: Optional[Dict] = None):
    """Build the model from Config(), apply optional overrides, and load weights.

    Returns
    -------
    model : the trained model in eval() mode on `device`
    cfg   : the Config object (post-shot-adaptive)
    """
    sys.path.insert(0, os.getcwd())
    train_mod = importlib.import_module(main_module)
    Config = train_mod.Config
    apply_shot = train_mod.apply_shot_adaptive_config
    Model = train_mod.MultiScaleMUOT_CONCH

    cfg = Config()
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    cfg = apply_shot(cfg)
    cfg.gpu_device_id = 0
    cfg.device = device

    model = Model(cfg).to(device)
    print(f"[load] reading checkpoint {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] WARN: {len(missing)} missing keys (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[load] WARN: {len(unexpected)} unexpected keys: {unexpected[:5]}")
    if hasattr(model, "memory") and model.memory is not None:
        try:
            model.memory.restore_initialized_flag_from_buffers()
        except Exception:
            pass
    model.eval()
    return model, cfg, train_mod


# ---------------------------------------------------------------------------
# Concept text features
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_concept_text_features(model) -> Dict[str, np.ndarray]:
    """Run the nested prompt learner once and pull concept-level outputs.

    Returns dict with:
        concept_names         : list[str]                (length C)
        class_associations    : list[str]                (length C)
        text_features         : np.ndarray (C, D)        L2-normalized
        diagnostic_prior      : np.ndarray (C,)          built-in concept_weights buffer
        learned_importance    : np.ndarray (C,) or None  sigmoid(swr.concept_importance)
                                                          if SWR is present
    """
    pl = model.nested_prompt_learner
    out = pl.forward(level='patch')
    text_feats = torch.cat(out['patch_features'], dim=0)  # (C, D)
    concept_names = list(out['tissue_names'])
    class_assoc = list(out['class_associations'])
    diag_prior = out['concept_weights'].detach().cpu().numpy()

    learned = None
    # Find an SWR layer if any aggregator is SemanticWassersteinAggregation
    for agg in model.scale_aggregators:
        if hasattr(agg, "swr_layer") and getattr(agg.swr_layer, "concept_importance", None) is not None:
            learned = torch.sigmoid(agg.swr_layer.concept_importance.detach()).cpu().numpy()
            break

    return {
        "concept_names":     concept_names,
        "class_associations": class_assoc,
        "text_features":     text_feats.detach().cpu().numpy().astype(np.float32),
        "diagnostic_prior":  diag_prior.astype(np.float32),
        "learned_importance": learned.astype(np.float32) if learned is not None else None,
        "_text_features_t":  text_feats.detach(),  # keep on device for direct use
    }


# ---------------------------------------------------------------------------
# Slide encoding
# ---------------------------------------------------------------------------
def _make_eval_transform():
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def encode_slide_at_scale(model, slide_item: Dict, scale: str,
                          max_patches: int = 1500,
                          transform=None) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """Run the visual encoder on a single slide at the given scale.

    Returns
    -------
    patch_features : (N, D=512)  L2-normalized
    patch_paths    : list[str] or None
                     Image paths in the SAME order as patch_features.
                     None when scale is feature-only (.pt files).
    """
    if transform is None:
        transform = _make_eval_transform()
    cfg = model.config
    device = cfg.device
    img_size = cfg.conch_img_size
    embed_dim = cfg.embedding_dim

    val = slide_item["scale_paths"][scale]
    scale_idx = cfg.scales.index(scale)

    # ----- (a) Pre-extracted features path -----
    if isinstance(val, list) and val and val[0].endswith(".pt"):
        feats = []
        for p in val[:max_patches]:
            feat = torch.load(p, map_location="cpu")
            if feat.dim() == 2 and feat.shape[0] == 1:
                feat = feat.squeeze(0)
            feats.append(feat)
        if not feats:
            return torch.zeros(0, embed_dim, device=device), None
        feats = torch.stack(feats, dim=0).to(device).float()
        feats = F.normalize(feats, dim=-1)
        return feats, None

    if isinstance(val, str) and val.endswith(".pt"):
        feat = torch.load(val, map_location="cpu")
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        feats = feat[:max_patches].to(device).float()
        feats = F.normalize(feats, dim=-1)
        return feats, None

    # ----- (b) Raw image path -----
    paths = list(val) if isinstance(val, list) else []
    if len(paths) > max_patches:
        idx = np.linspace(0, len(paths) - 1, max_patches, dtype=int).tolist()
        paths = [paths[i] for i in idx]

    images = []
    used_paths = []
    tgt = (img_size, img_size)
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.size != tgt:
                    im = im.resize(tgt, Image.Resampling.BILINEAR)
                images.append(transform(im))
                used_paths.append(p)
        except Exception:
            continue
    if not images:
        return torch.zeros(0, embed_dim, device=device), None

    pixels = torch.stack(images, dim=0).to(device)
    cls_emb, _ = model.encode_image_chunked(
        pixels, chunk_size=cfg.encoding_chunk_size,
        scale_idx=scale_idx, return_raw_spatial=False)
    cls_emb = F.normalize(cls_emb, dim=-1)
    return cls_emb, used_paths


# ---------------------------------------------------------------------------
# Transport plan
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_pi(model, patch_features: torch.Tensor,
               text_features: torch.Tensor,
               apply_swr_gates: bool = True,
               scale_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute UOT transport plan for one slide.

    Parameters
    ----------
    patch_features  : (N, D)  L2-normalized patch features
    text_features   : (C, D)  L2-normalized concept text features
    apply_swr_gates : if True (default), multiply by concept_weights
                      buffer and sigmoid(concept_importance) to mirror
                      what SWR actually uses inside the model.
    scale_idx       : which scale's SWR layer to read gates from (only
                      matters if multiple SWR aggregators exist)

    Returns
    -------
    pi_raw      : (N, C)  raw UOT plan (before gating)
    pi_effective: (N, C)  after the SWR gating that the model uses
    """
    if patch_features.numel() == 0:
        C = text_features.shape[0]
        return np.zeros((0, C), np.float32), np.zeros((0, C), np.float32)

    pf = patch_features.unsqueeze(0).contiguous()  # (1, N, D)
    tf = text_features.unsqueeze(0).contiguous()   # (1, C, D)

    _, pi = model.uot_intra(pf, tf, return_pi=True)  # (1, N, C)
    pi_raw = pi.squeeze(0).cpu().float().numpy()

    pi_eff = pi.squeeze(0)
    # gating: concept_weights buffer (prior)
    if hasattr(model.nested_prompt_learner, "concept_weights"):
        cw = model.nested_prompt_learner.concept_weights.to(pi_eff.device)
        pi_eff = pi_eff * cw.unsqueeze(0)
    # gating: learned concept_importance (if SWR present)
    if apply_swr_gates:
        for i, agg in enumerate(model.scale_aggregators):
            if hasattr(agg, "swr_layer") and getattr(agg.swr_layer, "concept_importance", None) is not None:
                if i == scale_idx:
                    ci = torch.sigmoid(agg.swr_layer.concept_importance.detach())
                    pi_eff = pi_eff * ci.to(pi_eff.device).unsqueeze(0)
                    break
    pi_eff = pi_eff.cpu().float().numpy()
    return pi_raw, pi_eff


# ---------------------------------------------------------------------------
# Slide indexing helpers
# ---------------------------------------------------------------------------
def load_split_with_paths(cfg, train_mod, split: str = "test") -> List[Dict]:
    """Re-index the dataset so we can attach raw patch paths back to each
    slide entry from the split JSON. Returns a list[dict] of slide items.
    """
    indexed = train_mod.index_rcc_dataset(
        cfg.rcc_root, cfg.class_dir_names, cfg.class_to_idx,
        scales=tuple(cfg.scales), min_patches_per_scale=cfg.scale_min_patches,
        conch_feats_root=cfg.conch_feats_root)
    splits_json = cfg.split_json
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"split JSON not found: {splits_json}")
    with open(splits_json) as f:
        d = json.load(f)
    slim = d[split]
    by_id = {it["slide_id"]: it for it in indexed}
    out = []
    for it in slim:
        sid = it["slide_id"]
        if sid in by_id:
            out.append(by_id[sid])
    print(f"[split] {split}: {len(out)} slides")
    return out


def pick_slides_for_class(split_data: List[Dict], num_classes: int = 3,
                          num_per_class: int = 2,
                          rng_seed: int = 0) -> List[Dict]:
    """Randomly pick K slides per class for visualization."""
    rng = np.random.RandomState(rng_seed)
    by_lbl: Dict[int, List[Dict]] = defaultdict(list)
    for it in split_data:
        by_lbl[int(it["label"])].append(it)
    out = []
    for c in range(num_classes):
        bucket = by_lbl.get(c, [])
        if not bucket:
            continue
        idx = rng.choice(len(bucket),
                         size=min(num_per_class, len(bucket)),
                         replace=False)
        for i in idx:
            out.append(bucket[int(i)])
    return out
