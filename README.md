# TFCM: Tiny Frequency Colorize Model

**A hybrid architecture for high-fidelity image colorization using Discrete Wavelet Transforms and MobileNetV3.**

| Input (Grayscale) | Output (TFCM Prediction) |
|:---:|:---:|
| ![Input Example](docs/coffee_BW.jpg) | ![Output Example](docs/coffee_C.jpg) |
| ![Input Example](docs/dog_BW.jpg) | ![Output Example](docs/dog_C.jpg) |

***image not from the COCO database**

## Overview

**TFCM** is an experimental deep learning model designed to solve the "washed out" color problem common in traditional CNNs. By processing images in the frequency domain, it separates structural details from semantic understanding.

This repository contains the training code, inference logic, and pretrained weights.

## Architecture

The core innovation of TFCM is its dual-branch topology:

![Architecture Diagram](docs/arch.png)

1.  **Semantic Branch (Blue):** Powered by a pre-trained **MobileNetV3 Large**. It analyzes the image content (context) to understand *what* the objects are (e.g., sky, grass, car) and generates a "Gating Map".
2.  **Structural Branch (Red):** Instead of standard pixel processing, this branch uses **Discrete Wavelet Transforms (DWT)**. Wavelets split the image into frequency bands, preserving high-frequency details (edges, textures) that are often lost in standard convolutions.
3.  **Fusion:** The semantic gate modulates the structural features (`Structure * (1 + Context)`), ensuring that colors are applied logically without blurring the fine details.


## Installation & Inference

### 1. Installation
Simply clone the repository and install via pip. The setup script handles dependencies.

```bash
git clone https://github.com/Juste-Leo2/TFCM.git
cd TFCM
pip install .
```

### 2. Inference
You do not need to hunt for weight files. The inference engine automatically detects if the model is missing and downloads the latest `tfcm_v14_large.pth` release from the repository.

**Python API:**
```python
from tfcm import Colorizer

# Initialize (downloads weights automatically)
model = Colorizer()

# Colorize
image = model.process("input_bw.jpg")
image.save("output.jpg")
```

**CLI:**
```bash
tfcm-cli --input my_photo.jpg --output result.jpg
```

## Training

To reproduce the results or train on your own dataset:

```bash
python download_coco.py
python train.py
```

## Dataset & Scalability

The current model was trained on the **COCO 2017 HD** dataset.
*   **Current Resolution:** Resized to 256x256 for this release.
*   **Scalability:** This architecture is highly promising for High-Resolution (4K) image processing. The DWT-based approach reduces spatial dimensions without losing information, making it computationally efficient for larger inputs compared to standard UNets.

## Training Results

The model exhibits stable convergence with minimal overfitting. The separation of structure and semantics allows the loss to decrease steadily.

![Loss Curve](docs/training_curve_v14.png)

## Installation & Inference

### 1. Installation
Simply clone the repository and install via pip. The setup script handles dependencies.

```bash
git clone https://github.com/Juste-Leo2/TFCM.git
cd TFCM
pip install .
```

### 2. Inference
You do not need to hunt for weight files. The inference engine automatically detects if the model is missing and downloads the latest `tfcm.pth` release from the repository.

**Python API:**
```python
from tfcm import Colorizer

# Initialize (downloads weights automatically)
model = Colorizer()

# Colorize
image = model.process("input_bw.jpg")
image.save("output.jpg")
```

**CLI:**
```bash
tfcm-cli --input my_photo.jpg --output result.jpg
```

## Training

To reproduce the results or train on your own dataset:

1.  Download **COCO 2017** (Train/Val).
2.  Configure paths in `train.py`.
3.  Run the training loop:

```bash
python train_v14_large.py
```

## License

This project is licensed under the **Apache License 2.0**.
You are free to use, modify, and distribute this software for personal or commercial use, provided you include the original copyright notice and license.