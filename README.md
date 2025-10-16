# Flow-Benchmarking
Overview

Morpho: flow encoder, +10% thickness edit, CF decode, metrics MAE, FD_attr(1D), CVS, LOC, SSIM(FCS), and FID.​

Causal: residual predictor on DAG (arm→blue→green→red), evaluate arm/blue/green/red with MAE, FFD, LOC, CVS, FCS.​

Setup

Python 3.9+, PyTorch, Torchvision; Inception‑v3 on CPU (299×299, ImageNet norm) for FID features.​

Data

Morpho: grayscale images + morpho CSVs with thickness (train/test).​

Causal: CSV per split with original_latents [[z_before],[z_after]], intervention_labels {0..4}, intervention_masks len 4.​

Run

Morpho: train flow, fit linear probe, run +10% edit and metrics (MAE/FD_attr/CVS/LOC/FCS/FID).​

Causal: train predictor, then evaluate_all_interventions for arm/blue/green/red.​

Outputs

Logs: CSV metrics and best checkpoints under logs/.​

Key formulas: FID uses Gaussian Fréchet with trace‑sqrt; MAE is mean absolute error.​
