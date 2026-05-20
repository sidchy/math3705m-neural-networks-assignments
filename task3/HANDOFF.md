# Task 3: Pix2Pix 灰度图像着色 — 交接文档

## 项目概述

课程"报告三：视觉模型的深度学习实践"，基于 GAN 的生成模型挑战。

选题：**基于 Pix2Pix 条件生成对抗网络的灰度图像自动着色**。

代码基于 [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 官方实现，使用 `--model colorization`（Lab 色彩空间方案）。训练在腾讯云 T4 GPU 上完成，报告用 LaTeX 写。

## 仓库

https://github.com/sidchy/math3705m-neural-networks-assignments

本地路径：`/Users/sidneychai/Documents/神经网络课程作业/math3705m-neural-networks-assignments`

## 文件位置

### LaTeX 报告（待编译）
```
task3/report/main.tex       # LaTeX 源文件（已写好）
task3/report/figures/       # 报告用图（已生成）
  ├── comparison_grid.png   # 8 例三行对比图（灰度/生成/真实）
  ├── example_1.png         # 单例对比
  ├── example_2.png         # 单例对比
  └── example_3.png         # 单例对比
```

### 自定义脚本
```
task3/scripts/preprocess_coco_colorization.py  # COCO 数据预处理
task3/scripts/evaluate_colorization.py         # MAE/PSNR/SSIM 评估
task3/train_coco.sh                            # 训练启动脚本
task3/test_coco.sh                             # 测试启动脚本
task3/setup_env.sh                             # uv 环境一键配置
task3/README.md                                # 项目说明
```

### 官方代码（fork 自 junyanz/pix2pix）
```
task3/models/colorization_model.py   # Lab 空间着色模型
task3/data/colorization_dataset.py   # RGB→Lab 自动转换
task3/train.py / test.py             # 训练/测试入口
```

### 结果文件
```
task3/results/colorization_pix2pix/test_latest/images/
  ├── *_real_A.png       # 灰度输入（L 通道）
  ├── *_fake_B_rgb.png   # 模型生成彩色图
  └── *_real_B_rgb.png   # 真实彩色图（ground truth）
```

### 模型权重（GitHub Releases: v3.0）
- `netG_stage1_fp32.pth`（208MB）— Stage 1, 30 epoch
- `netG_stage2_fp32.pth`（208MB）— Stage 2, 50 epoch
- 下载后放到 `task3/checkpoints/colorization_pix2pix/` 即可推理

## 实验结果

| Stage | Epoch | MAE | PSNR | SSIM |
|-------|-------|-----|------|------|
| 1 | 30 | 0.062 | 21.51 | 0.881 |
| 2 | 50 | 0.062 | 21.50 | 0.875 |

Stage 2 训练更久但 SSIM 反而略降，报告里解释为小样本下的过拟合。

## 当前状态

- ✅ 代码全部完成并推送 GitHub
- ✅ 模型训练完成（两个 stage）
- ✅ 对比图已生成
- ✅ LaTeX 报告已写好（main.tex）
- ⬜ 编译 LaTeX → PDF（需要 tectonic）
- ⬜ 模型权重上传 GitHub Releases（已下载到本地 Downloads）

## Codex 提示词

> 帮我编译 LaTeX 报告。报告在 `/Users/sidneychai/Documents/神经网络课程作业/math3705m-neural-networks-assignments/task3/report/main.tex`，用 tectonic 编译。如果没有 tectonic，先 `brew install tectonic`。编译命令：`cd task3/report && tectonic -X compile main.tex`。编译成功后把 PDF 路径告诉我。
>
> 然后，两个模型权重在 `/Users/sidneychai/Downloads/netG_stage1_fp32.pth` 和 `netG_stage2_fp32.pth`，帮我上传到 https://github.com/sidchy/math3705m-neural-networks-assignments/releases/new 作为 Release v3.0 的附件。上传后更新 `task3/README.md` 里的权重下载链接。
