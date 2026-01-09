#!/usr/bin/env python3
"""Stratified train/test split for image dataset.

Usage:
    python scripts/split_dataset.py --data-dir Data --out output --test-size 0.2

This script will copy files into `out/train/<class>` and `out/test/<class>`.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT and p.is_file()]


def gather_files(data_dir: Path):
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    classes.sort()
    items = []
    labels = []
    for c in classes:
        files = list_images(data_dir / c)
        for f in files:
            items.append(f)
            labels.append(c)
    return items, labels, classes


def copy_split(items, labels, classes, out_dir: Path, test_size: float, seed: int):
    out_dir = out_dir
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Convert to lists of strings (paths) for sklearn
    paths = [str(p) for p in items]
    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels, test_size=test_size, random_state=seed, stratify=labels if labels else None
    )

    for p, y in zip(X_train, y_train):
        dest = Path(p)
        target = train_dir / y
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, target / dest.name)

    for p, y in zip(X_test, y_test):
        dest = Path(p)
        target = test_dir / y
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, target / dest.name)

    # Print counts
    def count_files(folder: Path):
        return {c.name: len(list((folder / c.name).glob("*"))) if (folder / c.name).exists() else 0 for c in classes}

    print("Train counts:", count_files(train_dir))
    print("Test counts:", count_files(test_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data", help="Root data directory (contains class subfolders)")
    parser.add_argument("--out", default="output", help="Output folder for train/test split")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    items, labels, classes = gather_files(data_dir)
    if not items:
        print("No images found under", data_dir)
        return

    copy_split(items, labels, classes, Path(args.out), args.test_size, args.seed)


if __name__ == "__main__":
    main()
