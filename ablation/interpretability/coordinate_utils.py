"""
coordinate_utils.py
-------------------
Helpers for arranging WSI patches spatially:

1. parse_patch_coordinates(filenames)   — try to read (row, col) out of
   the filenames produced by common WSI patching pipelines (DSMIL-WSI,
   CLAM, Mahmood-Lab, custom scripts). Falls back to a square grid if
   nothing parses.

2. build_mosaic(patch_paths, coords, tile_size, max_dim)  — composite a
   thumbnail mosaic of all patches arranged at their grid coordinates.
   Used as the BASE image for transport-map heatmap overlays.

Filename patterns recognized
----------------------------
   17_42.jpeg                         → (17, 42)
   17_42.jpg                          → (17, 42)
   tile_17_42.png                     → (17, 42)
   patch_17_42.tiff                   → (17, 42)
   slide_TCGA-XX-XXXX_17_42.jpeg      → (17, 42)   (last two ints win)
   17_42_5x.jpeg                      → (17, 42)
   r17_c42.jpeg                       → (17, 42)
   x42_y17.jpeg                       → (17, 42)   (note: x→col, y→row)
   (17_42).jpeg                       → (17, 42)
"""

from __future__ import annotations
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------
# The patterns below run in priority order. The first to match wins.
_RC      = re.compile(r'r(\d+)[_-]c(\d+)', re.IGNORECASE)
_XY      = re.compile(r'x(\d+)[_-]y(\d+)', re.IGNORECASE)
# Generic "<int>_<int>" or "<int>x<int>" — last match wins. We deliberately
# require an underscore or 'x' separator (NOT hyphen), so hyphens inside
# TCGA barcodes ("TCGA-XX-XXXX") don't get caught.
_INT_INT = re.compile(r'(\d+)[_x](\d+)')


def _parse_one(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (row, col) from a single basename. Returns None on failure.

    Supported patterns
    ------------------
       r17_c42.*                     → (17, 42)
       x42_y17.*                     → (17, 42)   (x↔col, y↔row)
       17_42.*                       → (17, 42)
       tile_17_42.*                  → (17, 42)
       patch_17_42.*                 → (17, 42)
       slide_TCGA-XX-XXXX_17_42.*    → (17, 42)
       17x42.*                       → (17, 42)
    """
    stem = Path(filename).stem  # remove extension

    m = _RC.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = _XY.search(stem)
    if m:
        col = int(m.group(1))
        row = int(m.group(2))
        return row, col

    # Last <int>_<int> (or <int>x<int>) pair in the stem. We use re.findall
    # then take the last to handle filenames that have multiple integer
    # pairs (e.g. "TCGA_12_3456_17_42" → grabs "17_42").
    pairs = _INT_INT.findall(stem)
    if pairs:
        a, b = pairs[-1]
        return int(a), int(b)

    return None


def parse_patch_coordinates(filenames: List[str]) -> Tuple[np.ndarray, str]:
    """Try to parse all filenames; return (coords, mode).

    coords  : np.array shape (N, 2) of (row, col).
    mode    : 'parsed' if filenames yielded coords, 'grid' if we had to
              fall back to a square grid layout.
    """
    out = []
    for fn in filenames:
        rc = _parse_one(fn)
        out.append(rc)
    valid = [rc for rc in out if rc is not None]
    if len(valid) >= max(2, int(0.6 * len(filenames))):
        # Replace any missing entries with a synthetic position
        side = int(math.ceil(math.sqrt(len(filenames))))
        for i, rc in enumerate(out):
            if rc is None:
                out[i] = (i // side, i % side)
        coords = np.array(out, dtype=int)
        return coords, 'parsed'

    # Pure fallback: square grid
    n = len(filenames)
    side = int(math.ceil(math.sqrt(n)))
    coords = np.array([(i // side, i % side) for i in range(n)], dtype=int)
    return coords, 'grid'


# ---------------------------------------------------------------------------
# Mosaic builder
# ---------------------------------------------------------------------------
def normalise_coords(coords: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Shift coords to be 0-indexed; return (coords, n_rows, n_cols)."""
    if coords.size == 0:
        return coords, 0, 0
    coords = coords.copy()
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    n_rows = int(coords[:, 0].max()) + 1
    n_cols = int(coords[:, 1].max()) + 1
    return coords, n_rows, n_cols


def build_mosaic(
    patch_paths: List[str],
    coords: np.ndarray,
    tile_size: int = 48,
    max_dim: int = 2400,
    background: Tuple[int, int, int] = (245, 245, 245),
) -> Tuple[Image.Image, int, int, int]:
    """Build a thumbnail mosaic from patch image paths and their coords.

    Returns
    -------
    mosaic     : PIL.Image  (RGB)
    tile_size  : tile size actually used (may have been auto-shrunk)
    n_rows     : grid rows
    n_cols     : grid cols
    """
    coords, n_rows, n_cols = normalise_coords(coords)
    if n_rows == 0 or n_cols == 0:
        return Image.new("RGB", (1, 1), background), tile_size, 0, 0

    # Auto-shrink tile_size to keep the canvas under max_dim
    canvas_w = n_cols * tile_size
    canvas_h = n_rows * tile_size
    if max(canvas_w, canvas_h) > max_dim:
        scale = max_dim / max(canvas_w, canvas_h)
        tile_size = max(8, int(tile_size * scale))
        canvas_w = n_cols * tile_size
        canvas_h = n_rows * tile_size

    canvas = Image.new("RGB", (canvas_w, canvas_h), background)

    for path, (r, c) in zip(patch_paths, coords):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB").resize(
                    (tile_size, tile_size), Image.Resampling.BILINEAR)
                canvas.paste(im, (c * tile_size, r * tile_size))
        except Exception:
            continue
    return canvas, tile_size, n_rows, n_cols


def values_to_grid(values: np.ndarray, coords: np.ndarray,
                   n_rows: int, n_cols: int,
                   fill: float = np.nan) -> np.ndarray:
    """Project a 1-D vector of per-patch values into a 2-D grid using coords.

    Cells without a patch are set to `fill`.
    """
    grid = np.full((n_rows, n_cols), fill, dtype=np.float32)
    for v, (r, c) in zip(values, coords):
        if 0 <= r < n_rows and 0 <= c < n_cols:
            grid[r, c] = v
    return grid
