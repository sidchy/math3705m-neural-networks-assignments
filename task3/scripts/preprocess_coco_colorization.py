#!/usr/bin/env python3
"""Preprocess COCO val2017 images for Pix2Pix colorization.

Converts raw COCO images into a simple RGB dataset split into train/val/test.
The official ColorizationDataset handles RGB->Lab conversion internally.

Usage:
    python scripts/preprocess_coco_colorization.py \
        --coco_root ./coco \
        --out_root ./datasets/colorization \
        --image_size 256 \
        --train 1600 --val 200 --test 200
"""

import argparse
import os
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO colorization dataset")
    parser.add_argument("--coco_root", type=str, required=True,
                        help="Path to COCO directory (containing val2017/ folder)")
    parser.add_argument("--out_root", type=str, default="./datasets/colorization",
                        help="Output directory for processed dataset")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize image to image_size x image_size")
    parser.add_argument("--train", type=int, default=1600,
                        help="Number of training images")
    parser.add_argument("--val", type=int, default=200,
                        help="Number of validation images")
    parser.add_argument("--test", type=int, default=200,
                        help="Number of test images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()

    # Paths
    coco_root = Path(args.coco_root)
    val_dir = coco_root / "val2017"
    if not val_dir.is_dir():
        # Try train2017 as fallback
        val_dir = coco_root / "train2017"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"COCO image directory not found. Looked for:\n"
            f"  {coco_root / 'val2017'}\n"
            f"  {coco_root / 'train2017'}\n"
            f"Please download COCO val2017 first."
        )

    out_root = Path(args.out_root)
    image_size = args.image_size

    # Collect all image paths
    image_paths = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {val_dir}")

    total_needed = args.train + args.val + args.test
    if total_needed > len(image_paths):
        print(f"WARNING: Only {len(image_paths)} images available, "
              f"but {total_needed} requested. Using all images.")
        # Rebalance
        ratio = len(image_paths) / total_needed
        args.train = int(args.train * ratio)
        args.val = int(args.val * ratio)
        args.test = len(image_paths) - args.train - args.val
        total_needed = len(image_paths)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(image_paths)
    selected = image_paths[:total_needed]
    train_paths = selected[:args.train]
    val_paths = selected[args.train:args.train + args.val]
    test_paths = selected[args.train + args.val:]

    # Process each split
    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(paths, desc=f"Processing {split_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = resize_and_crop(img, image_size)
                out_path = out_dir / f"{img_path.stem}.png"
                img.save(out_path, "PNG")
            except Exception as e:
                print(f"WARNING: Skipping {img_path.name}: {e}")

    print(f"\nDataset ready at {out_root.resolve()}")
    print(f"  train: {len(train_paths)} images")
    print(f"  val:   {len(val_paths)} images")
    print(f"  test:  {len(test_paths)} images")


def resize_and_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize so shortest side = size, then center crop to size x size."""
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


if __name__ == "__main__":
    main()
