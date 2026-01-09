#!/usr/bin/env python3
"""Small EDA for the Malaria image dataset.

Usage:
    python scripts/eda.py --data-dir Data --samples 5 --out output/eda.png

This script samples a few images per class, plots them in a grid and
prints simple class counts and image size statistics.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT and p.is_file()]


def sample_images(data_dir: Path, samples_per_class: int = 5):
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    classes.sort()

    sampled = {}
    counts = {}
    sizes = {c: [] for c in classes}

    for c in classes:
        files = list_images(data_dir / c)
        counts[c] = len(files)
        if files:
            sampled[c] = random.sample(files, min(samples_per_class, len(files)))
            for p in files:
                try:
                    with Image.open(p) as im:
                        sizes[c].append(im.size)
                except Exception:
                    pass
        else:
            sampled[c] = []

    return classes, counts, sizes, sampled


def plot_grid(classes, sampled, counts, sizes, out_path: Path | None = None):
    # Determine grid size
    max_cols = max(len(v) for v in sampled.values()) if sampled else 0
    rows = len(classes)
    if rows == 0 or max_cols == 0:
        print("No images found to plot.")
        return

    fig, axes = plt.subplots(rows, max_cols, figsize=(max(3 * max_cols, 6), 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for i, c in enumerate(classes):
        for j in range(max_cols):
            ax = axes[i, j]
            ax.axis("off")
            try:
                p = sampled[c][j]
            except IndexError:
                continue
            try:
                im = Image.open(p).convert("RGB")
                ax.imshow(im)
                ax.set_title(f"{c}\n{p.name}")
            except Exception:
                ax.text(0.5, 0.5, "error", ha="center", va="center")

    plt.suptitle("Sample images per class")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved sample grid to {out_path}")
    else:
        plt.show()

    # Print counts and simple size stats
    print("\nClass counts:")
    for c, cnt in counts.items():
        print(f"  {c}: {cnt}")

    print("\nImage size stats (width x height) per class (first 5 shown):")
    for c, ss in sizes.items():
        print(f"  {c}: {len(ss)} images; sample sizes: {ss[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data", help="Root data directory (contains class subfolders)")
    parser.add_argument("--samples", type=int, default=5, help="Samples per class to plot")
    parser.add_argument("--out", default=None, help="Output path to save the figure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    classes, counts, sizes, sampled = sample_images(data_dir, samples_per_class=args.samples)
    out_path = Path(args.out) if args.out else None
    plot_grid(classes, sampled, counts, sizes, out_path)


if __name__ == "__main__":
    main()
