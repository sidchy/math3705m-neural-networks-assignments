#!/usr/bin/env python3
"""Evaluate trained colorization model with MAE, PSNR, SSIM metrics.

Computes metrics between generated RGB (fake_B_rgb) and ground truth RGB (real_B_rgb).
Results are computed per-image and averaged.

Usage:
    python scripts/evaluate_colorization.py \
        --dataroot ./datasets/colorization \
        --name colorization_pix2pix \
        --epoch latest
"""

import argparse
import os
import sys

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epoch", type=str, default="latest")
    parser.add_argument("--num_test", type=int, default=0,
                        help="0 means use all test images")
    args = parser.parse_args()

    # Add project root to path so we can import data/model modules
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from data import create_dataset
    from models import create_model
    from options.test_options import TestOptions

    # Build options mimicking test.py
    sys.argv = [
        "evaluate",
        "--dataroot", args.dataroot,
        "--name", args.name,
        "--model", "colorization",
        "--netG", "unet_256",
        "--preprocess", "none",
        "--phase", "test",
        "--epoch", args.epoch,
        "--num_test", str(args.num_test),
    ]

    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # create_dataset returns an iterable CustomDatasetDataLoader (already
    # wraps a DataLoader internally — do NOT wrap it again)
    dataset = create_dataset(opt)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    mae_list = []
    psnr_list = []
    ssim_list = []

    for i, data in enumerate(tqdm(dataset, desc="Evaluating")):
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        fake_rgb = visuals["fake_B_rgb"]  # (H, W, 3) numpy uint8 [0, 255]
        real_rgb = visuals["real_B_rgb"]

        fake_f = np.clip(fake_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        real_f = np.clip(real_rgb.astype(np.float32) / 255.0, 0.0, 1.0)

        mae = np.mean(np.abs(fake_f - real_f))
        psnr = peak_signal_noise_ratio(real_f, fake_f, data_range=1.0)

        h, w = fake_f.shape[:2]
        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        ssim = structural_similarity(
            real_f, fake_f, channel_axis=2, data_range=1.0, win_size=win_size
        )

        mae_list.append(mae)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(f"\n=== Evaluation Results ({len(mae_list)} test images) ===")
    print(f"MAE:   {np.mean(mae_list):.4f}  (+/- {np.std(mae_list):.4f})")
    print(f"PSNR:  {np.mean(psnr_list):.2f}  (+/- {np.std(psnr_list):.2f}) dB")
    print(f"SSIM:  {np.mean(ssim_list):.4f}  (+/- {np.std(ssim_list):.4f})")


if __name__ == "__main__":
    main()
