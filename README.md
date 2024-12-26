# MultiStyle-Neural-Transfer


This project explores **Neural Style Transfer (NST)** using various deep learning models to merge artistic styles with content images. By analyzing the performance of multiple architectures and loss functions, it identifies the most effective approaches for NST.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Models and Performance](#models-and-performance)
- [Dataset](#dataset)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Introduction
Neural Style Transfer (NST) is a technique to create visually compelling images by transferring the style of an artwork onto a content image. This project:
- Employs multiple architectures (e.g., VGG19, ResNet50V2, ResNeXt-50, and DenseNet-121).
- Benchmarks the performance using metrics like **PSNR** (Peak Signal-to-Noise Ratio) and **content preservation scores**.
- Utilizes the **WikiArt dataset** for training and evaluation.

---

## Key Features
- Comparison of **four different models** for NST:
  - **VGG19 with MSE Loss**: Baseline model using Gram matrices.
  - **ResNet50V2**: Leveraging residual connections for efficient gradient flow.
  - **ResNeXt-50**: Superior performance with high PSNR and content preservation scores.
  - **DenseNet-121**: Balances style rendering and content integrity.
- Integration of **Adaptive Gram Matrix Loss** to dynamically adjust style influences.
- Detailed performance analysis of each model.

---

## Models and Performance
| Model                     | PSNR (dB) | Notes                                                                 |
|---------------------------|-----------|-----------------------------------------------------------------------|
| **VGG19 (Baseline)**      | 16.41     | Straightforward Gram matrices and MSE loss; computationally intensive.|
| **DenseNet-121**          | 15.87     | Efficient but introduces stylization noise.                          |
| **VGG19 (Adaptive Loss)** | 18.07     | Dynamic loss adjustments; excels in aesthetic quality.               |

---

## Dataset
The project leverages the **WikiArt dataset**, a comprehensive collection of artwork, to train and evaluate the models. The dataset is used to:
- Extract artistic styles.
- Test content fidelity across different NST approaches.

---


