# Pruning-Aware Tokenization in Vision Transformers

<p align="center">
  <img src="figures/diagram.png" width="850"/>
</p>

Architecture of the Two-Scale Tokenization Stem

## Overview

We show that replacing the standard tokenizer with a small, multi-layer convolutional stem that introduces hierarchical, nonlinear feature extraction 
and enlarged receptive fields can improve the robustness of ViTs to early token pruning. 
Furthermore, we extend this design with a secondary branch that generates a supplementary set of low-resolution tokens. 
By providing greater spatial coverage, this coarse-scale representation allows a small subset of tokens to preserve a global view of the image even under aggressive pruning rates.

## Key Results

<p align="center">
  <img src="figures/results.png" width="850"/>
</p>


| Model | Accuracy | GFLOPs | Throughput |
|------|----------|--------|------------|
| ViT baseline | ... | ... | ... |
| PAT-ViT | ... | ... | ... |

## Method

[architecture]

## Token Pruning

[visualization]

## Experiments

### ImageNet-1K

...

### Ablation Studies

...

## Installation

...

## Training

...

## Evaluation

...

## Citation

...
