# CIFAR-10 Image Generation with Diffusion Models

Denoising and image-generation experiments using PyTorch diffusion models. The project explores the progression from noisy inputs to generated images, with image-space and latent-space modelling code and tools for inspecting learned representations.

## Model components

The source includes DDPM, conditional diffusion, latent diffusion and conditional latent-diffusion branches. Noise-prediction networks include U-Net and encoder/decoder variants. Configuration selects the active branch; the presence of a class does not mean every combination uses the same trained model or results.

The fixed-run settings in `dummy_config.yaml` select CIFAR-10, 32 × 32 RGB images and conditional diffusion. The separate study configuration contains different image-shape and naming settings; the two launchers should not be assumed to describe an identical experiment.

## Source map

| Location | Purpose |
|---|---|
| `Networks/ddpm.py` | Image-space diffusion model |
| `Networks/latent_diffusion.py` | Latent-space diffusion model |
| `Networks/unet.py` and `Networks/unet_latent.py` | Noise-prediction networks |
| `Networks/encoder_decoder.py` | Encoder/decoder alternatives |
| `data_loader.py` | Dataset preparation |
| `learner.py` and `learner_latent.py` | Training and evaluation |
| `utils.py` | Sampling and auxiliary model loading |
| `plots.py` | Image, representation and attribution plots |
| `dummy_main.py` | Fixed run using `dummy_config.yaml` |
| `main.py` and `optuna_hyp.py` | Hyperparameter study |

## Preparing an experiment

The code uses PyTorch and the external `ccbdl` framework for configuration, data loading, experiment storage and parts of the learning workflow. A compatible installation of that framework is required; it is not bundled here. Other dependencies include torchvision, NumPy, Matplotlib, Optuna and Captum, with additional analysis libraries used by individual modules.

Use the original compatible environment, prepare the dataset at the configured location, and run from the repository root so relative paths resolve correctly. The archive does not include a complete dependency lock file. Supply datasets and optional pretrained models separately where referenced.

Review the dataset, channels, image size, model branch, noise-prediction network, noise schedule and sampling settings together. Latent experiments need the encoder/decoder checkpoint expected by the selected path. Classifier/discriminator analysis has additional pretrained-model requirements.

After preparation, the fixed-run entry point is:

```bash
python dummy_main.py
```

This reads `dummy_config.yaml`; the study launcher reads `config.yaml`. Both filenames are retained from the original experiments.

## Outputs and interpretation

Plotting routines cover noisy-image sequences, generated grids, loss curves and representation/attribution analysis. Assess sampling quality alongside diversity, configuration and the number of denoising steps.

Auxiliary feature-space distances depend on the feature extractor. A Fréchet distance computed with a custom classifier is not directly comparable with a standard Inception-feature FID benchmark. PSNR and SSIM need meaningful image pairing; they do not alone measure the quality of unrelated generated samples.

## Scope

This repository preserves research code and configurations. It does not assert an independently verified benchmark score or identical setup requirements for every branch. Original dataset terms and external framework/model attribution remain applicable.
