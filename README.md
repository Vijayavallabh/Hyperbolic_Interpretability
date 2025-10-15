# Sparse Hyperbolic Convolutional Networks with Enhanced Object Localization via GradCAM Analysis

**ICCV 2025 | Beyond Euclidean**

## Overview

This repository offers a unified framework for sparse interpretability in deep neural networks, supporting:

- **Euclidean ResNet:** Standard convolutional neural networks (CNNs)
- **Hyperbolic ResNet:** Networks leveraging Lorentz/hyperbolic geometry
- **Hybrid ResNet:** Architectures combining Euclidean and hyperbolic blocks

**Key Features:**

- k-Sparse and L1 sparsity constraints for interpretable feature learning
- GradCAM-based hierarchical interpretability across all geometries
- Comprehensive interpretability evaluation metrics: robustness, faithfulness, localization, complexity, and randomisation
- Modular and extensible codebase for rapid experimentation with new architectures or geometries


## Introduction

Sparse interpretability enhances neural network transparency by enforcing sparsity (e.g., k-sparse, L1) in feature activations and visualising influential regions via GradCAM. This repository extends these interpretability techniques to hyperbolic geometry (Lorentz model) and hybrid architectures, providing deeper insights into hierarchical and compositional representations in deep learning.

## Features

**Sparsity Constraints**

- L1 and k-sparse regularisation for both Euclidean and Lorentz layers
- Block-wise and layer-wise sparsity statistics for detailed analysis

**Interpretability**

- GradCAM support for Euclidean, Lorentz, and Hybrid ResNets
- Hierarchical interpretability analysis across network depth
- k-sparse-aware GradCAM for sparsity-constrained models

**Evaluation Metrics**

- Robustness to input noise
- Faithfulness (removal/insertion metrics)
- Localization (IoU, entropy, effective area)
- Complexity and randomisation assessments

**Modular Design**

- Seamless switching between Euclidean, Lorentz, and Hybrid models
- Configuration via command-line arguments or config files


## Installation

Clone the repository:

```bash
git clone https://github.com/Vijayavallabh/hyperbolic-interpretability.git
cd hyperbolic-interpretability
```

Install dependencies:

```bash
conda create --name venv python=3.10
conda activate venv
pip install -r requirements.txt
```



## Usage

```bash
chmod +x bash_cell.sh
nohup ./bash_cell.sh > output.log 2>&1 &

```

### Training

Train a model with your chosen sparsity and geometry:

**Euclidean ResNet**

```bash
python code/classification/train.py -c classification/config/E-ResNet18.txt --output_dir classification/output --device cuda:0
```

**Hyperbolic (Lorentz) ResNet**

```bash
python code/classification/train.py -c classification/config/L-ResNet18.txt --output_dir classification/output --device cuda:0
```

**Hybrid ResNet**

```bash
python code/classification/train.py -c classification/config/EL-ResNet18.txt --output_dir classification/output --device cuda:0
```

To enable k-sparse constraints, append:

```
--enable_sparsity --sparsity_type k_sparse --k_ratio 0.1
```


### Evaluation \& Interpretability

```bash
python code/classification/interpretability.py -c classification/config/L-ResNet18.txt --model_path PATH/TO/WEIGHTS.pth --analysis_type evaluation --output_dir classification/output
```



## Citation

If you use this code or ideas in your research, please cite:

```
@inproceedings{
jayamanikandan2025sparse,
title={Sparse Hyperbolic Convolutional Networks with Enhanced Object Localization via Grad{CAM} Analysis},
author={Vijayavallabh Jayamanikandan and Settur Jithamanyu and Lokesh Kumar Rajulapati and Raghunathan Rengaswamy},
booktitle={2nd Beyond Euclidean Workshop: Hyperbolic and Hyperspherical Learning for Computer Vision},
year={2025},
url={https://openreview.net/forum?id=Y9DWlKCUbK}
}
```


## License

This code is released under the MIT License.

## Acknowledgements

- **HyperbolicCV:** Hyperbolic geometry tools


For questions or contributions, please open an issue or pull request.


