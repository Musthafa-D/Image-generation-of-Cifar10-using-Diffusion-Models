# CIFAR-10 Image Generation Using Diffusion Models (PyTorch)

## Project Overview

This project implements **image generation on the CIFAR-10 dataset using diffusion models** in PyTorch.  
Diffusion models are powerful generative models that learn image generation through a gradual forward noise diffusion process and a reverse denoising process.

This repository contains code to train and evaluate different diffusion models for generating CIFAR-10 images and visualize the results and also to interpret the trained models using various interpretability methods such as attributions, metrics, etc.

---

## Dataset

- **CIFAR-10**
- 60,000 32×32 color images
- 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Both conditional and unconditional of denoising diffusion models as well as latent diffsuion models are trained and evaluated.

---

## What This Project Includes

- PyTorch implementation of a diffusion model
- Dataset preprocessing and augmentation
- Training loop for forward diffusion and reverse denoising
- Image sampling / generation
- Visualization of generated images over training steps
- Interpretability of the trained models.

---

