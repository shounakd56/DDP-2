from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("extract_plip_tcga_20x")


# -------------------------------------------------------------------
# Expected dataset structure:
#
# /home/datasets/tcga_brca/
#     <slide_id>/
#         20x/
#             patch_1.png
#             patch_2.jpg
#         5x/
#             patch_1.png
#
# This script extracts ONLY 20x PLIP features.
# -------------------------------------------------------------------

DEFAULT_INPUT_ROOT = "/home/datasets/tcga_brca"
DEFAULT_OUTPUT_ROOT = "/home/datasets/tcga_brca_plip_feats_20x"

DEFAULT_IMG_SIZE = 224
PLIP_MODEL_NAME = "vinid/plip"

VALID_EXT = {".jpeg", ".jpg", ".png"}


def collect_patches(
    input_root: Path,
    out_root: Path,
    magnification: str = "20x",
    skip_existing: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Collect image patches only from:
        /home/datasets/tcga_brca/<slide_id>/20x/

    Output:
        /home/datasets/tcga_brca_plip_feats_20x/<slide_id>/20x/<patch>.pt
    """

    pairs: List[Tuple[Path, Path]] = []

    slide_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    logger.info(f"Found {len(slide_dirs):,} slide directories")

    missing_mag_dirs = 0

    for slide_dir in slide_dirs:
        mag_dir = slide_dir / magnification

        if not mag_dir.exists() or not mag_dir.is_dir():
            missing_mag_dirs += 1
            continue

        for img_path in sorted(mag_dir.rglob("*")):
            if not img_path.is_file():
                continue

            if img_path.suffix.lower() not in VALID_EXT:
                continue

            rel = img_path.relative_to(input_root)
            out_path = (out_root / rel).with_suffix(".pt")

            if skip_existing and out_path.exists():
                continue

            pairs.append((img_path, out_path))

    if missing_mag_dirs > 0:
        logger.warning(
            f"{missing_mag_dirs:,} slide folders did not contain a '{magnification}' folder"
        )

    return pairs


class PatchDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        processor: CLIPProcessor,
        img_size: int,
    ):
        self.pairs = pairs
        self.processor = processor
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        in_path, out_path = self.pairs[idx]

        try:
            with Image.open(in_path) as img:
                img = img.convert("RGB")

                if img.size != (self.img_size, self.img_size):
                    img = img.resize(
                        (self.img_size, self.img_size),
                        Image.Resampling.BILINEAR,
                    )

                inputs = self.processor(
                    images=img,
                    return_tensors="pt",
                )

                pixel_values = inputs["pixel_values"].squeeze(0)

            ok = True

        except Exception as e:
            logger.warning(f"[load-fail] {in_path}: {e}")
            pixel_values = torch.zeros(3, self.img_size, self.img_size)
            ok = False

        return pixel_values, str(in_path), str(out_path), ok


def collate(batch):
    pixel_values = torch.stack([b[0] for b in batch], dim=0)
    in_paths = [b[1] for b in batch]
    out_paths = [b[2] for b in batch]
    oks = [b[3] for b in batch]

    return pixel_values, in_paths, out_paths, oks


def load_plip(device: torch.device, local_files_only: bool = False):
    logger.info(f"Loading PLIP model: {PLIP_MODEL_NAME}")

    processor = CLIPProcessor.from_pretrained(
        PLIP_MODEL_NAME,
        local_files_only=local_files_only,
    )

    model = CLIPModel.from_pretrained(
        PLIP_MODEL_NAME,
        local_files_only=local_files_only,
    )

    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    logger.info(f"PLIP loaded on {device}")

    return model, processor


@torch.inference_mode()
def encode_batch(
    model: CLIPModel,
    pixel_values: torch.Tensor,
    use_amp: bool,
) -> torch.Tensor:
    """
    Robust PLIP image feature extraction.

    Do NOT use model.get_image_features() here, because in some PLIP /
    transformers installs it can return BaseModelOutputWithPooling instead
    of a tensor.

    This explicitly runs:
        vision_model -> pooler_output -> visual_projection -> L2 normalize
    """

    if use_amp and pixel_values.is_cuda:
        with autocast("cuda", dtype=torch.float16):
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            pooled_output = vision_outputs.pooler_output
            feats = model.visual_projection(pooled_output)
    else:
        vision_outputs = model.vision_model(
            pixel_values=pixel_values.float(),
            return_dict=True,
        )
        pooled_output = vision_outputs.pooler_output
        feats = model.visual_projection(pooled_output)

    feats = feats.float()
    feats = F.normalize(feats, p=2, dim=-1)

    return feats


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract PLIP features from TCGA 20x patches"
    )

    p.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing slide folders",
    )

    p.add_argument(
        "--out_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to save .pt PLIP embedding files",
    )

    p.add_argument(
        "--magnification",
        type=str,
        default="20x",
        help="Magnification folder to process. Default: 20x",
    )

    p.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--gpu", type=int, default=3)

    p.add_argument(
        "--fp16",
        action="store_true",
        help="Store embeddings in float16 instead of float32",
    )

    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision during inference",
    )

    p.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Overwrite existing .pt files",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N images. 0 means all.",
    )

    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load PLIP only from local Hugging Face cache",
    )

    return p.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not input_root.exists():
        logger.error(f"Input root does not exist: {input_root}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input root      : {input_root}")
    logger.info(f"Output root     : {out_root}")
    logger.info(f"Magnification   : {args.magnification}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

        if args.gpu >= n_gpus:
            logger.warning(
                f"Requested cuda:{args.gpu}, but only {n_gpus} CUDA device(s) "
                f"available. Using cuda:0."
            )
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected. Using CPU.")

    logger.info(f"Device          : {device}")
    logger.info("Scanning dataset...")

    pairs = collect_patches(
        input_root=input_root,
        out_root=out_root,
        magnification=args.magnification,
        skip_existing=not args.no_skip_existing,
    )

    if args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        logger.info("Nothing to do. No images found or all embeddings already exist.")
        return

    logger.info(f"Images to process: {len(pairs):,}")

    for _, out_path in pairs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_plip(
        device=device,
        local_files_only=args.local_files_only,
    )

    dataset = PatchDataset(
        pairs=pairs,
        processor=processor,
        img_size=args.img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    out_dtype = torch.float16 if args.fp16 else torch.float32
    use_amp = device.type == "cuda" and not args.no_amp

    n_done = 0
    n_fail = 0

    for pixel_values, in_paths, out_paths, oks in tqdm(
        loader,
        desc="Encoding 20x PLIP patches",
        unit="batch",
    ):
        pixel_values = pixel_values.to(device, non_blocking=True)

        feats = encode_batch(
            model=model,
            pixel_values=pixel_values,
            use_amp=use_amp,
        )

        feats = feats.to(out_dtype).cpu()

        for i in range(feats.shape[0]):
            if not oks[i]:
                n_fail += 1
                continue

            try:
                torch.save(feats[i].clone(), out_paths[i])
                n_done += 1
            except Exception as e:
                n_fail += 1
                logger.warning(f"[save-fail] {out_paths[i]}: {e}")

    logger.info("Finished.")
    logger.info(f"Saved embeddings : {n_done:,}")
    logger.info(f"Failed images    : {n_fail:,}")
    logger.info(f"Output root      : {out_root}")


if __name__ == "__main__":
    main()