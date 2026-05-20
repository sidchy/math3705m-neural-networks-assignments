# 课程报告三：基于 Pix2Pix 的灰度图像自动着色

基于 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 官方实现，使用 COCO 2017 子集训练 Pix2Pix 模型完成灰度图像自动着色任务。

## 环境配置

```bash
pip install torch torchvision dominate visdom scikit-image tqdm Pillow
```

验证 GPU：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 数据准备

1. 下载 COCO 2017 val2017（约 1GB）：
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./coco/
```

2. 预处理：
```bash
# 时间紧（2000 张）
python scripts/preprocess_coco_colorization.py \
  --coco_root ./coco --out_root ./datasets/colorization \
  --image_size 256 --train 1600 --val 200 --test 200

# 完整（5000 张）
python scripts/preprocess_coco_colorization.py \
  --coco_root ./coco --out_root ./datasets/colorization \
  --image_size 256 --train 4000 --val 500 --test 500
```

## 训练

```bash
bash train_coco.sh stage1   # 20+10 epoch, 快速验证
bash train_coco.sh stage2   # 30+20 epoch, 正式模型
bash train_coco.sh stage3   # 50+50 epoch, 扩展训练
```

模型权重保存在 `checkpoints/colorization_pix2pix/`。

## 测试

```bash
bash test_coco.sh            # 使用 latest checkpoint
bash test_coco.sh 20         # 使用第 20 epoch 权重
```

输出三列对比图到 `results/colorization_pix2pix/test_latest/`。

## 评估

```bash
python scripts/evaluate_colorization.py \
  --dataroot ./datasets/colorization \
  --name colorization_pix2pix --epoch latest
```

输出 MAE、PSNR、SSIM。

## 模型权重

训练好的 `.pth` 文件下载：[网盘链接]

放到 `checkpoints/colorization_pix2pix/` 后运行测试脚本即可推理。

## 项目结构

```
task3_gan_colorization/
├── train.py / test.py           # 官方入口
├── models/                       # 官方模型（ColorizationModel 等）
├── data/                         # 官方数据加载
│
├── scripts/                      # 自定义脚本
│   ├── preprocess_coco_colorization.py
│   └── evaluate_colorization.py
├── train_coco.sh                 # 训练脚本
├── test_coco.sh                  # 测试脚本
│
├── datasets/colorization/        # 数据（不上传 Gitee）
├── checkpoints/                  # 权重（不上传 Gitee）
├── results/                      # 推理结果
└── report/                       # LaTeX 实验报告
```

## 引用

- Isola et al. Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.
- Zhu et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV 2017.
