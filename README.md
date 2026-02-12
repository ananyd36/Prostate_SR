
# ProstaSR: Deep Generative 3D Super-Resolution for Prostate MRI

Magnetic Resonance Imaging (MRI) of the prostate is typically acquired with high in-plane resolution (0.5 × 0.5 mm) but low through-plane resolution (3.0 mm) to reduce scan time. When reformatted in 3D, this anisotropy introduces staircase artifacts that degrade segmentation, registration, and diagnostic quality.

**ProstaSR** addresses this problem by framing reconstruction as a **Through-Plane Multi-Image Super-Resolution (MISR)** task:

> Predict the missing slice *z* using its neighboring slices *(z−1, z+1)*.

This repository implements and benchmarks deterministic CNNs, GANs, and Diffusion models to analyze pixel-level accuracy vs perceptual realism in medical image interpolation.

---

## Dataset

* **Prostate-MRI-US-Biopsy Dataset (TCIA)**
* 842 patient volumes
* T2-weighted axial sequences only
* Preprocessing:

  * 1st–99th percentile intensity clipping
  * Normalization to [0,1]
  * Resized to 256×256
  * Slice triplet generation
  * Safe DICOM validation loader

Split:

* 70% Train
* 15% Validation
* 15% Test

---

## Implemented Architectures

### 1️⃣ SRCNN (Baseline Regression)

* 3-layer CNN
* Large kernel receptive fields (9×9, 5×5)
* Optimized with L1 loss
* Maximizes pixel-level accuracy (PSNR)

---

### 2️⃣ Deep-SRCNN (Residual Learning)

* Deeper 3×3 convolution stack
* Increased non-linearity
* Improved texture modeling
* Still optimized via L1 regression

---

### 3️⃣ SRGAN (Adversarial Reconstruction)

* U-Net Generator

* Patch-based Discriminator

* Composite Loss:

  L_total = L_adv + λ L_content (λ = 100)

* Recovers sharper textures

* Improves perceptual realism

* Risk of minor artifacts

---

### 4️⃣ Fast-DDPM (Conditional Diffusion)

To overcome regression over-smoothing and GAN instability:

* Conditional Residual U-Net
* Noise prediction objective
* Time-aware residual blocks with sinusoidal embeddings
* Accelerated inference:

  * Compressed diffusion schedule
  * 1000 → 10 denoising steps
  * Clinically viable runtime


---

## Experimental Setup

* PyTorch
* NVIDIA L4 GPU
* Mixed Precision (AMP)
* AdamW (lr = 1e−4)

Metrics:

* **PSNR ↑**
* **SSIM ↑**
* **MAE ↓**

---

## Quantitative Results

| Model             | PSNR (dB) | SSIM  | MAE    |
| ----------------- | --------- | ----- | ------ |
| SRCNN             | 26.09     | 0.839 | 0.0475 |
| Deep-SRCNN        | 26.20     | 0.841 | 0.0466 |
| SRGAN             | 25.12     | 0.831 | 0.0502 |
| Fast-DDPM (15 ep) | 19.89     | 0.604 | 0.0874 |

---

## Key Findings

* **CNNs maximize PSNR** but produce over-smoothed anatomical boundaries.
* **SRGAN improves structural realism (SSIM ~0.83)** with sharper capsule and lesion boundaries.
* **Fast-DDPM**, even with limited training (15 epochs), shows strong potential for modeling complex high-frequency anatomical textures.
* Clear trade-off between pixel accuracy and perceptual fidelity.

---

## Visual Comparison

Each test sample includes:

* Context Slices (z−1, z+1)
* Ground Truth (z)
* Predictions from:

  * SRCNN
  * Deep-SRCNN
  * SRGAN
  * Fast-DDPM

Generative models recover higher-frequency details compared to regression-based smoothing.

---

## Future Work

* Extend diffusion training to 100+ epochs
* Integrate attention in bottleneck layers
* Explore perceptual feature losses
* Compare 2D vs 3D convolutional approaches
* Evaluate downstream segmentation impact

---

## Research Context

This project was developed as part of graduate-level coursework in Deep Learning for Medical Imaging at the University of Florida.

It builds upon:

* SRCNN (Dong et al., TPAMI 2016)
* VDSR (Kim et al., CVPR 2016)
* SRGAN (Ledig et al., CVPR 2017)
* DDPM (Ho et al., NeurIPS 2020)
* Fast-DDPM for medical imaging (JBHI 2025)

---


## Author

Anany Sharma <br>
M.S. Artificial Intelligence Systems<br>
University of Florida<br>

---
