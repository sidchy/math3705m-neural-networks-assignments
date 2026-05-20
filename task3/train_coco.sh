#!/bin/bash
# Train Pix2Pix colorization model on COCO grayscale->color task.
# Uses official --model colorization (Lab color space, auto RGB<->Lab conversion).
#
# Usage:
#   bash train_coco.sh          # Stage 1: 20+10 epochs (default)
#   bash train_coco.sh stage2   # Stage 2: 30+20 epochs
#   bash train_coco.sh stage3   # Stage 3: 50+50 epochs
#
# The ColorizationDataset reads plain RGB images from:
#   datasets/colorization/train/
# and converts them to (L, ab) pairs internally.

set -ex

STAGE="${1:-stage1}"

case "$STAGE" in
  stage1)
    N_EPOCHS=20
    N_EPOCHS_DECAY=10
    echo "=== Stage 1: Quick validation (20+10 epochs) ==="
    ;;
  stage2)
    N_EPOCHS=30
    N_EPOCHS_DECAY=20
    echo "=== Stage 2: Formal model (30+20 epochs) ==="
    ;;
  stage3)
    N_EPOCHS=50
    N_EPOCHS_DECAY=50
    echo "=== Stage 3: Extended training (50+50 epochs) ==="
    ;;
  *)
    echo "Usage: bash train_coco.sh [stage1|stage2|stage3]"
    exit 1
    ;;
esac

python train.py \
  --dataroot ./datasets/colorization \
  --name colorization_pix2pix \
  --model colorization \
  --netG unet_256 \
  --netD basic \
  --preprocess none \
  --batch_size 1 \
  --n_epochs "${N_EPOCHS}" \
  --n_epochs_decay "${N_EPOCHS_DECAY}" \
  --display_freq 100 \
  --print_freq 50 \
  --save_epoch_freq 5

echo "Training complete. Checkpoints saved to checkpoints/colorization_pix2pix/"
