#!/usr/bin/env python3
"""
Ablation runner.
Imports from train.py and replaces the OT modules to test:
- full_uot (baseline)
- balanced_ot
- no_ot (cosine pooling)
- component‑wise mixtures.

Usage:
    python ablation.py --k 16 --ablation_mode full_uot [--uot_tau 0.5]
"""

import os
import sys
import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms          # <-- added
from collections import defaultdict                  # <-- added
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
# --------------------------------------------------------------------
# Ensure train.py is importable from the current directory
# --------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from your original train.py
from train import (
    Config,
    apply_shot_adaptive_config,
    # OT
    UnbalancedOptimalTransport as _OriginalUOT,
    # Main model
    MultiScaleMUOT_CONCH as _OriginalModel,
    # Helpers
    build_routing_concepts,
    logger,
    # Dataset & collate
    index_rcc_dataset,
    make_patient_level_splits,
    save_splits_to_json,
    load_splits_from_json,
    attach_paths_to_split,
    MultiScaleWSIDataset,
    FilteredMemoryDataset,
    collate_multiscale_batch,
    create_balanced_sampler,
    create_few_shot_subset,
    # Evaluation
    evaluate_wsi,
    evaluate_test_full_report,
    # Training
    train_one_epoch,
    # Checkpoint
    save_checkpoint,
    load_checkpoint,
    # SWA
    SWAModel,
    # Stain
    StainAugmentation,
    # Tokenizer
    _safe_tokenize,
)

# --------------------------------------------------------------------
# 1. Ablation Config – extends original Config with extra fields
# --------------------------------------------------------------------
class AblationConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ablation_mode = 'full_uot'      # full_uot, balanced_ot, no_ot, uot_swr_bal_mem, bal_swr_uot_mem
        self.uot_tau_override = None         # if set, override tau_a and tau_b
        # Ensure save_dir reflects ablation
        self.save_dir = None

# --------------------------------------------------------------------
# 2. Override UnbalancedOptimalTransport to handle balanced mode
# --------------------------------------------------------------------
class AblationUOT(_OriginalUOT):
    def __init__(self, tau_a=1.0, tau_b=1.0, epsilon=0.1, max_iter=50,
                 adaptive_stop=True, convergence_thresh=1e-4, balanced=False):
        super().__init__(tau_a, tau_b, epsilon, max_iter, adaptive_stop, convergence_thresh)
        self.balanced = balanced

    def solve_uot_stabilized(self, C, a, b):
        if self.balanced:
            # Force uniform marginals → balanced Sinkhorn
            B, M, N = C.shape
            device = C.device
            eps = self.epsilon + 1e-10
            # Uniform marginals
            a = torch.ones(B, M, device=device, dtype=torch.float32) / M
            b = torch.ones(B, N, device=device, dtype=torch.float32) / N
            log_u = torch.zeros(B, M, device=device, dtype=torch.float32)
            log_v = torch.zeros(B, N, device=device, dtype=torch.float32)
            log_K = -C / eps
            for it in range(self.max_iter):
                log_u = -torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2) - math.log(M)
                log_v = -torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2) - math.log(N)
                if self.adaptive_stop and it > 5 and it % 5 == 0:
                    # optional early stop – omitted for brevity
                    pass
            pi = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
            d = torch.sum(pi * C, dim=(1, 2))
            d = torch.where(torch.isnan(d) | torch.isinf(d),
                            torch.tensor(2.0, device=device), d)
            return torch.clamp(d, 0.0, 2.0), pi
        else:
            # Standard unbalanced Sinkhorn (unchanged)
            return super().solve_uot_stabilized(C, a, b)

# --------------------------------------------------------------------
# 3. Subclass the main model to inject ablation OT modules
# --------------------------------------------------------------------
class AblationModel(_OriginalModel):
    def __init__(self, config):
        # Let the original __init__ do most of the work
        super().__init__(config)
        # Re‑build OT modules according to ablation_mode
        self.uot_swr, self.uot_mem = self._build_ablation_ots(config)
        # Overwrite the original uot_intra (used by SWR aggregator)
        self.uot_intra = self.uot_swr

    def _build_ablation_ots(self, config):
        mode = config.ablation_mode
        tau_a = config.uot_tau_a
        tau_b = config.uot_tau_b
        if config.uot_tau_override is not None:
            tau_a = tau_b = config.uot_tau_override

        eps = config.uot_epsilon
        max_iter = config.uot_max_iter
        adaptive = config.uot_adaptive_stop
        thresh = config.uot_convergence_thresh

        # Balanced OT: use huge tau to simulate strict mass preservation
        bal_tau = 1e4

        if mode == 'full_uot':
            uot_swr = AblationUOT(tau_a=tau_a, tau_b=tau_b, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=False)
            uot_mem = AblationUOT(tau_a=tau_a, tau_b=tau_b, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=False)
        elif mode == 'balanced_ot':
            uot_swr = AblationUOT(tau_a=bal_tau, tau_b=bal_tau, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=True)
            uot_mem = AblationUOT(tau_a=bal_tau, tau_b=bal_tau, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=True)
        elif mode == 'no_ot':
            uot_swr = None
            uot_mem = None
        elif mode == 'uot_swr_bal_mem':
            uot_swr = AblationUOT(tau_a=tau_a, tau_b=tau_b, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=False)
            uot_mem = AblationUOT(tau_a=bal_tau, tau_b=bal_tau, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=True)
        elif mode == 'bal_swr_uot_mem':
            uot_swr = AblationUOT(tau_a=bal_tau, tau_b=bal_tau, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=True)
            uot_mem = AblationUOT(tau_a=tau_a, tau_b=tau_b, epsilon=eps,
                                  max_iter=max_iter, adaptive_stop=adaptive,
                                  convergence_thresh=thresh, balanced=False)
        else:
            raise ValueError(f"Unknown ablation_mode: {mode}")
        return uot_swr, uot_mem

    # ------------------------------------------------------------------
    # Helper to compute OT or cosine distance depending on module existence
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_ot_distance(self, X, Y, module):
        if module is None:
            # Cosine distance: 1 – mean(cosine similarity)
            Xn = F.normalize(X.float(), dim=-1)
            Yn = F.normalize(Y.float(), dim=-1)
            d = 1 - torch.bmm(Xn, Yn.transpose(1, 2)).mean(dim=(1, 2))
            return torch.clamp(d, 0.0, 2.0)
        else:
            return module(X, Y, return_pi=False)

    # ------------------------------------------------------------------
    # Full forward_wsi with the only modification: memory distance uses self.uot_mem
    # (the rest is identical to the original v3.1 code, so we replicate it here)
    # ------------------------------------------------------------------
    def forward_wsi(self, multi_patches, labels=None, multi_patch_counts=None):
        if not isinstance(multi_patches, dict):
            sn = self.scale_names[0]
            multi_patches = {sn: multi_patches}
            multi_patch_counts = {sn: multi_patch_counts}
        if multi_patch_counts is None:
            multi_patch_counts = {}

        B = list(multi_patches.values())[0].shape[0]
        device = list(multi_patches.values())[0].device

        text_features = self.nested_prompt_learner(level='both')
        concept_weights = text_features.get('concept_weights', None)

        swr_text_features = None
        if 'patch_features' in text_features and self._use_pl:
            tissue_stacked = torch.cat(text_features['patch_features'], dim=0)
            swr_text_features = tissue_stacked.unsqueeze(0).expand(B, -1, -1)

        scale_slide_features = []
        memory_spatial = None
        memory_patch_feat = None
        memory_patch_counts = None
        memory_pertile_spatial = None

        for s_idx, sn in enumerate(self.scale_names):
            if sn not in multi_patches:
                continue
            is_mem = (s_idx == self.memory_scale_idx)
            scale_text = (swr_text_features if self._swr_scale_flags[s_idx] else None)

            (slide_feat, patch_feat, slide_spatial,
             pertile_spatial, updated_counts) = self._encode_scale(
                multi_patches[sn], multi_patch_counts.get(sn, None), s_idx,
                return_spatial=is_mem,
                text_features=scale_text,
                concept_weights=concept_weights)

            scale_slide_features.append(slide_feat)

            if is_mem:
                memory_spatial = slide_spatial
                memory_patch_feat = patch_feat
                memory_patch_counts = updated_counts
                memory_pertile_spatial = pertile_spatial
            else:
                del patch_feat, slide_spatial, pertile_spatial

        fused_slide = F.normalize(self.scale_fusion(scale_slide_features), dim=-1)

        d_patch_class = None
        if (self._use_pl and 'patch_features' in text_features
                and memory_patch_feat is not None and hasattr(self, 'tissue_mapper')):
            concept_stacked = torch.cat(text_features['patch_features'], dim=0)
            sim = torch.einsum('bnd,td->bnt',
                               F.normalize(memory_patch_feat, dim=-1),
                               F.normalize(concept_stacked, dim=-1))
            _, N_s, _ = sim.shape
            if memory_patch_counts is not None:
                mask = (torch.arange(N_s, device=sim.device).unsqueeze(0)
                        < memory_patch_counts.unsqueeze(1)).unsqueeze(-1).expand_as(sim)
                ca = (sim * mask.float()).sum(dim=1) / mask.sum(dim=1).float().clamp(min=1.0)
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
            d_slide = torch.clamp(
                1 - torch.matmul(F.normalize(fused_slide, dim=-1),
                                 F.normalize(class_stacked, dim=-1).T), 0.0, 2.0)

        # ---- Memory distance (ABLATION EDIT: using self.uot_mem) ----
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
                            d = self._compute_ot_distance(memory_spatial, bary_batch,
                                                          self.uot_mem)
                        except Exception:
                            d = 1 - torch.bmm(
                                F.normalize(memory_spatial, dim=-1),
                                F.normalize(bary_batch, dim=-1).transpose(1, 2)
                            ).mean(dim=(1, 2))
                            d = torch.clamp(d, 0.0, 2.0)
                        proto_dists.append(d)
                    proto_stack = torch.stack(proto_dists, dim=1)
                    wk = self.memory.prototype_weights[k].unsqueeze(0).to(device)
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
            mu = torch.sigmoid(self.mu_slide)
            d_combined = mu * d_slide + (1 - mu) * d_I
        else:
            d_combined = d_I

        if d_patch_class is not None:
            av = torch.sigmoid(self.alpha)
            d_final = av * d_combined + (1 - av) * d_patch_class
        else:
            d_final = d_combined

        if d_direct is not None:
            dw = torch.sigmoid(self.direct_weight)
            d_final = (1 - dw) * d_final + dw * d_direct

        probs = self.compute_probability_from_distance(d_final)

        if labels is not None:
            smooth = self.config.label_smoothing
            n_cls = self.config.num_classes
            st = torch.full_like(probs, smooth / n_cls)
            st.scatter_(1, labels.unsqueeze(1), 1.0 - smooth + smooth / n_cls)
            loss_ce = -(st * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            cp = -0.1 * entropy
            bi = torch.arange(B, device=device)

            def margin_loss(d, labs):
                if d is None:
                    return torch.tensor(0.0, device=device)
                dp = d[bi, labs]
                m = torch.ones_like(d, dtype=torch.bool)
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
                total_loss = total_loss + cp + pdl
            else:
                total_loss = (loss_ce + 0.1 * lm_s + 0.1 * lm_I
                              + 0.05 * lm_p + 0.05 * lm_d + cp + pdl)
                lw = [1.0, 0.1, 0.1, 0.05, 0.05]

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


# --------------------------------------------------------------------
# 4. Override shot‑adaptive config to keep ablation fields
# --------------------------------------------------------------------
def apply_shot_adaptive_config_ablation(cfg):
    # Call original shot‑adaptive settings
    cfg = apply_shot_adaptive_config(cfg)
    # Ensure save_dir includes ablation mode and k
    base = "./fewshot_ckpt_conch_rcc_k{}_20x_full_seed_109".format(cfg.few_shot_k_train)
    if cfg.ablation_mode != 'full_uot':
        cfg.save_dir = base + "_" + cfg.ablation_mode
    else:
        cfg.save_dir = base
    if cfg.uot_tau_override is not None:
        cfg.save_dir += f"_tau{cfg.uot_tau_override}"
    return cfg


# --------------------------------------------------------------------
# 5. Main training function adapted for ablation
# --------------------------------------------------------------------
def run_ablation(args):
    # Build config from ablation arguments
    cfg = AblationConfig()
    cfg.few_shot_k_train = args.k
    cfg.ablation_mode = args.ablation_mode
    if args.uot_tau is not None:
        cfg.uot_tau_override = args.uot_tau
    cfg.rebuild_splits = args.rebuild_splits
    cfg.resume = args.resume
    cfg.resume_from = args.resume_from
    cfg.num_epochs = args.epochs if args.epochs is not None else cfg.num_epochs

    # Apply shot‑adaptive config and finalise save_dir
    cfg = apply_shot_adaptive_config_ablation(cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION MODE: {cfg.ablation_mode}")
    logger.info(f"UOT tau override: {cfg.uot_tau_override}")
    logger.info(f"Save directory: {cfg.save_dir}")
    logger.info(f"{'='*60}\n")

    # ---- 1. Index dataset ----
    all_indexed = index_rcc_dataset(
        cfg.rcc_root, cfg.class_dir_names, cfg.class_to_idx,
        scales=tuple(cfg.scales), min_patches_per_scale=cfg.scale_min_patches,
        conch_feats_root=cfg.conch_feats_root)
    if not all_indexed:
        logger.error("No slides indexed.")
        return

    # ---- 2. Build splits ----
    if cfg.rebuild_splits or not os.path.exists(cfg.split_json):
        train_full, val_full, test_full = make_patient_level_splits(
            all_indexed, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
            seed=cfg.split_seed, num_classes=cfg.num_classes)
        save_splits_to_json(train_full, val_full, test_full, cfg.split_json)
    else:
        train_slim, val_slim, test_slim = load_splits_from_json(cfg.split_json)
        train_full = attach_paths_to_split(train_slim, all_indexed)
        val_full = attach_paths_to_split(val_slim, all_indexed)
        test_full = attach_paths_to_split(test_slim, all_indexed)

    # Few‑shot subset
    train_data = (create_few_shot_subset(train_full, cfg.few_shot_k_train)
                  if cfg.few_shot_train else train_full)

    # ---- 3. Data loaders ----
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)
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

    train_ds = MultiScaleWSIDataset(train_data, tt, True, cfg)
    val_ds = MultiScaleWSIDataset(val_full, vt, False, cfg)
    test_ds = MultiScaleWSIDataset(test_full, vt, False, cfg)

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

    # ---- 4. Model (ablation version) ----
    model = AblationModel(cfg).to(cfg.device)
    tp = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {tp:,} total, {tr:,} trainable ({tr/tp*100:.2f}%)")

    # ---- 5. Optimiser, scheduler, scaler ----
    gc_map = {
        'visual_encoder': ('vp', 0.5),
        'nested_prompt_learner.routing_soft_ctx': ('rc', 1.0),
        'nested_prompt_learner.slide_soft_ctx': ('sc', 1.0),
        'scale_aggregators': ('ag', 2.0),
        'scale_fusion': ('fu', 2.0),
        'tissue_mapper': ('mp', 2.0),
    }
    grp = defaultdict(list)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        matched = False
        for k, (g, _) in gc_map.items():
            if k in n:
                grp[g].append(p)
                matched = True
                break
        if not matched:
            grp['other'].append(p)

    glr = {'vp': 0.5, 'rc': 1.0, 'sc': 1.0, 'ag': 2.0, 'fu': 2.0, 'mp': 2.0, 'other': 0.1}
    pg = [{'params': ps, 'lr': cfg.learning_rate * glr.get(g, 0.1)} for g, ps in grp.items()]
    opt = torch.optim.AdamW(pg, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(10, cfg.num_epochs // 4), T_mult=2, eta_min=cfg.min_lr)
    scaler = GradScaler() if cfg.mixed_precision else None
    swa_model = SWAModel(model) if cfg.swa_enabled else None

    # ---- 6. Resume or init memory ----
    last_ckpt_path = os.path.join(cfg.save_dir, "last_checkpoint.pth")
    start_epoch = 0
    best_acc = 0.0
    best_bal_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = defaultdict(list)

    if cfg.resume and os.path.exists(last_ckpt_path):
        info = load_checkpoint(last_ckpt_path, model, opt, scheduler, scaler,
                               swa_model, device=cfg.device)
        start_epoch = info['epoch'] + 1
        best_acc = info['best_acc']
        best_bal_acc = info['best_bal_acc']
        best_val_loss = info['best_val_loss']
        patience_counter = info['patience_counter']
        history = info['history']
        logger.info(f"Resumed from epoch {info['epoch']}")
        if model.memory is not None and not model.memory.memory_initialized:
            # Rebuild memory if missing
            mem_ds = FilteredMemoryDataset(train_ds.data, cfg.memory_scale, vt,
                                           img_size=cfg.conch_img_size)
            mem_dl = DataLoader(mem_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
            model.init_memory(mem_dl)
    else:
        # Fresh training – initialise memory
        mem_ds = FilteredMemoryDataset(train_ds.data, cfg.memory_scale, vt,
                                       img_size=cfg.conch_img_size)
        mem_dl = DataLoader(mem_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        model.init_memory(mem_dl)

    # ---- 7. Training loop ----
    for epoch in range(start_epoch, cfg.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        train_metrics = train_one_epoch(model, tl, opt, scaler, cfg, epoch)
        scheduler.step()
        for k, v in train_metrics.items():
            history[k].append(v)

        # Validation
        if (epoch + 1) % cfg.val_frequency == 0 or epoch == cfg.num_epochs - 1:
            val_results = evaluate_wsi(model, vl, cfg)
            history['val_acc'].append(val_results['acc'])
            history['val_bal_acc'].append(val_results['balanced_acc'])
            history['val_auc'].append(val_results['auc'])
            history['val_f1'].append(val_results['f1'])
            history['val_loss'].append(val_results['val_loss'])
            history['val_epoch'].append(epoch+1)
            logger.info(f"Val Acc={val_results['acc']:.4f} BalAcc={val_results['balanced_acc']:.4f}")

            # Update bests
            if val_results['acc'] > best_acc:
                best_acc = val_results['acc']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'best_acc': best_acc, 'best_bal_acc': best_bal_acc,
                            'history': dict(history)},
                           os.path.join(cfg.save_dir, "best_acc_model.pth"))
            if val_results['balanced_acc'] > best_bal_acc:
                best_bal_acc = val_results['balanced_acc']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'best_acc': best_acc, 'best_bal_acc': best_bal_acc,
                            'history': dict(history)},
                           os.path.join(cfg.save_dir, "best_bal_acc_model.pth"))
            if val_results['val_loss'] < best_val_loss:
                best_val_loss = val_results['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if (epoch >= cfg.min_train_epochs and cfg.early_stopping_patience > 0
                    and patience_counter >= cfg.early_stopping_patience):
                logger.info("Early stopping triggered.")
                break

        save_checkpoint(last_ckpt_path, model, opt, scheduler, scaler,
                        epoch, best_acc, best_bal_acc, best_val_loss,
                        patience_counter, history, swa_model)

    # ---- 8. Final test evaluation ----
    best_path = os.path.join(cfg.save_dir, "best_bal_acc_model.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(cfg.save_dir, "best_acc_model.pth")
    if os.path.exists(best_path) and test_full:
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f"Loaded best model for test evaluation.")
        evaluate_test_full_report(model, test_dl, cfg,
                                  os.path.join(cfg.save_dir, "test_results"))

    # Save history and config
    with open(os.path.join(cfg.save_dir, "history.json"), 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v]
                   for k, v in history.items()}, f, indent=2)
    with open(os.path.join(cfg.save_dir, "config_used.json"), 'w') as f:
        json.dump({k: v for k, v in vars(cfg).items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, type(None)))},
                  f, indent=2)

    logger.info(f"Ablation {cfg.ablation_mode} finished. Best Acc={best_acc:.4f}, Best BalAcc={best_bal_acc:.4f}")
    return {'best_acc': best_acc, 'best_bal_acc': best_bal_acc}


# --------------------------------------------------------------------
# 6. Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=16, help='Number of shots per class')
    parser.add_argument('--ablation_mode', type=str, default='full_uot',
                        choices=['full_uot', 'balanced_ot', 'no_ot',
                                 'uot_swr_bal_mem', 'bal_swr_uot_mem'])
    parser.add_argument('--uot_tau', type=float, default=None,
                        help='Override tau_a and tau_b (for full_uot only usually)')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--rebuild_splits', action='store_true', help='Regenerate data splits')
    args = parser.parse_args()

    torch.manual_seed(54)
    np.random.seed(54)
    import random
    random.seed(54)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(54)
        torch.cuda.manual_seed_all(54)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_ablation(args)