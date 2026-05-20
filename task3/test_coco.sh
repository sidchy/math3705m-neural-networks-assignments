#!/bin/bash
# Test Pix2Pix colorization model and generate comparison visualizations.
#
# The official ColorizationModel outputs:
#   real_A       - L channel input (grayscale)
#   real_B_rgb   - ground truth RGB (converted from Lab)
#   fake_B_rgb   - predicted RGB (converted from Lab)
#
# Results saved to results/colorization_pix2pix/test_latest/

set -ex

EPOCH="${1:-latest}"

python test.py \
  --dataroot ./datasets/colorization \
  --name colorization_pix2pix \
  --model colorization \
  --netG unet_256 \
  --preprocess none \
  --phase test \
  --eval \
  --no_flip \
  --num_test 50 \
  --epoch "${EPOCH}"

echo "Results saved to results/colorization_pix2pix/test_${EPOCH}/"
