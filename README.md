# Sparse Neural Network Training and Optimization

This project studies sparse neural network training on CIFAR-10 with a ResNet-20 style model. It compares fixed-masking and dynamic sparsity strategies under multiple sparsity levels and optimizers to understand when sparse training can preserve accuracy.

## Project Summary

Modern neural networks are often heavily over-parameterized. The core question in this project is whether a model can keep competitive performance while training with substantially fewer active weights.

We evaluate three sparsity strategies:

- Static Sparse Training: a fixed binary mask applied at initialization and kept unchanged throughout training.
- Sparse Evolutionary Training (SET): sparse connections are randomly pruned and regrown over time.
- RigL (Rigging the Lottery): sparse connections are updated using gradient information to regrow important weights.

Each method is evaluated at 90%, 95%, and 98% sparsity with both SGD with momentum and Adam.

## Repository Layout

```text
.
├── 00_baseline.ipynb        # Dense baseline experiment
├── 01_task_1.ipynb         # Static sparse training
├── 01_task_2.ipynb         # SET dynamic sparsity
├── 01_task_3.ipynb         # RigL dynamic sparsity
├── Final_Paper.pdf         # Final report
├── models.py               # ResNetCIFAR model and residual blocks
├── data/
│   └── cifar-10-batches-py/ # CIFAR-10 data files
├── results/                # Saved experiment artifacts
├── img/                    # Generated figures used in the report
└── README.md
```

## Methods

### Static Sparse Training

A binary mask is sampled once at initialization and remains fixed for the full training run. This gives a simple baseline for sparse learning.

### Sparse Evolutionary Training (SET)

SET maintains a sparse network by removing connections and randomly regrowing new ones during training.

### RigL

RigL also maintains constant sparsity, but regrowth is guided by gradient magnitude so the model can reallocate capacity more effectively.

## Main Observations

- Static sparsity is a strong baseline and remains surprisingly competitive at moderate sparsity.
- RigL generally performs better than SET, showing the value of gradient-guided regrowth.
- Optimizer choice matters: Adam is typically more stable for dynamic sparsity, while SGD is stronger in some static settings.
- Performance drops sharply at 98% sparsity, suggesting the model begins to lose too much capacity.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

Install the core dependencies with:

```bash
pip install torch torchvision matplotlib numpy
```

## How To Run

Run the notebooks in sequence:

1. `00_baseline.ipynb` for the dense baseline.
2. `01_task_1.ipynb` for static sparse training.
3. `01_task_2.ipynb` for SET.
4. `01_task_3.ipynb` for RigL.

Each notebook trains the model, logs accuracy and loss, and generates plots or artifacts used in the final report.

## Model Details

- Architecture: `ResNetCIFAR`, a CIFAR-style ResNet-20 variant defined in `models.py`.
- Dataset: CIFAR-10.
- Data augmentation: not used.
- Training epochs: 15.
- Batch size: 128.
- Optimizers:
  - SGD with momentum 0.9 and learning rate 0.1.
  - Adam with learning rate 0.001.

## Outputs

The repository includes the following generated outputs:

- Training and test accuracy curves.
- Final accuracy versus sparsity plots.
- Weight/kernel visualizations.
- Connectivity evolution plots for RigL.
- Saved run artifacts in `results/`.

## Notes

- The repository follows a `task_n_*` naming pattern for saved artifacts in `results/` and `img/`.
- `00_baseline.ipynb` imports `ResNetCIFAR` from `models.py` instead of redefining the model in the notebook.

## Authors

- Ammy Lin
- Tonantzin Real Rojas
- Anvita Suresh

## Course Context

This repository was created for ECE 685D/CS 675D: Introduction to Deep Learning at Duke University with Professor Vahid Tarokh.