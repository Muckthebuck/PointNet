# PointNet Implementation (WIP)

This repository contains a clean, from-scratch implementation of **PointNet**, the deep learning architecture introduced in the paper:

> [**PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**](https://arxiv.org/abs/1612.00593)  
> Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas  
> Stanford University, 2017

## Overview

PointNet is a pioneering neural network architecture that directly consumes unordered 3D point clouds for classification and segmentation tasks. It leverages symmetry functions and max-pooling to maintain invariance to input permutation.

This implementation is intended for educational and research purposes and aims to be:

- Faithful to the original architecture  
- Clean and readable  
- Easy to extend or modify  

## Requirements

- CUDA: 12.1


Install dependencies using conda

```bash
conda env create -f environment.yml
conda activate pointnet_env
```

The project uses pytorch3d chamfer distance, but if pytorch3d cant be used (windows) then you can utilise the implementation provided in utils.loss. However, its not recommended as its not optimised for large batches. s