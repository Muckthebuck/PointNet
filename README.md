# PointNet Implementation 

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

Install dependencies using pip:

```bash
pip install -r requirements.txt
```
