# Improving Music Source Separation with Diffusion and Consistency Refinement

<p align="center"></p>

This repository houses the official PyTorch implementation of the paper **"Improving Music Source Separation with Diffusion and Consistency Refinement"**.

- [arXiv](https://arxiv.org/abs/2412.06965)
- [Demo](https://consistency-separation.github.io/)

> An earlier version of this work, **"Improving Source Extraction with Diffusion and Consistency Models"**, was presented as an oral at the [NeurIPS 2024 Workshop on Audio Imagination](https://neurips.cc/virtual/2024/105752).

**Contacts**:
- Tornike Karchkhadze: [tkarchkhadze@ucsd.edu](mailto:tkarchkhadze@ucsd.edu)
- Mohammad Rasool Izadi: [russell_izadi@bose.com](mailto:russell_izadi@bose.com)

*Part of this work was done during Tornike's internship at Bose Corporation.*

---

## Abstract

In this work, we propose an approach to music source separation that uses a generative diffusion model as a last-stage refinement on top of a deterministic separator, progressively enhancing the separated sources through iterative denoising. While the diffusion refinement yields measurable quality gains, it requires iterative steps at inference, increasing computational cost. To speed up the inference process, we apply consistency distillation, reducing inference to a single step while maintaining quality; with two or more steps, the distilled model even surpasses the diffusion-based approach. Crucially, our method is architecture-agnostic: we demonstrate state-of-the-art results when applied to both a custom U-Net-based separator on Slakh2100 and the state-of-the-art BS-RoFormer model on MUSDB18, showing that the refinement generalizes across backbone architectures.

---

## Prerequisites

### 1. Datasets

This project uses two datasets:

- **Slakh2100** (U-Net experiments): Please follow the instructions for data download and setup provided here:
  [Slakh2100 Data Setup](https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md)

  ```bash
  mkdir -p dataset/slakh2100
  ```

- **MUSDB18-HQ** (BS-RoFormer experiments): Download directly from Zenodo:

  ```bash
  # Create the dataset directory
  mkdir -p dataset/musdb18hq

  # Download MUSDB18-HQ (approximately 30 GB)
  wget https://zenodo.org/record/3338373/files/musdb18hq.zip

  # Unzip
  unzip musdb18hq.zip -d dataset/musdb18hq

  # Remove the zip file
  rm musdb18hq.zip
  ```

### 2. Conda Environment Setup

This repository uses Python 3.9.19.

```bash
# Create environment
conda env create -f environment.yaml

# Activate environment
conda activate dicose
```

### 3. Weights & Biases

Training and evaluation both require a [Weights & Biases](https://wandb.ai) account for logging.

```bash
wandb login
```

---

## Training

### U-Net Experiments (Slakh2100)

#### Deterministic Model Training
```bash
python train_audio_simple.py --cfg configs/deterministic_model/unet_train.yaml
```

#### Diffusion Model Training
```bash
python train_audio.py --cfg configs/diffusion_model/unet_train.yaml
```

#### Consistency Model Training
```bash
python main_audio_ctm.py --cfg configs/consistency_model/unet_train.yaml
```

### BS-RoFormer Experiments (MUSDB18)

Code coming soon!

---

## Sampling and Evaluation

### Checkpoints

All pre-trained checkpoints are hosted on Hugging Face Hub: [karchkha/DiCoSe](https://huggingface.co/karchkha/DiCoSe).

```bash
hf download karchkha/DiCoSe --local-dir lightning_logs/DiCoSe
```

Or download a single checkpoint, e.g.:

```bash
hf download karchkha/DiCoSe CD_unet/model.ckpt --local-dir lightning_logs/DiCoSe
```

Each checkpoint lands at `lightning_logs/DiCoSe/<name>/model.ckpt`.

### U-Net Experiments (Slakh2100)

#### Deterministic Model Evaluation
```bash
python train_audio_simple.py --cfg configs/deterministic_model/unet_eval.yaml
```

#### Diffusion Model Evaluation
```bash
python train_audio.py --cfg configs/diffusion_model/unet_eval.yaml
```

#### Consistency Model Evaluation
```bash
python main_audio_ctm.py --cfg configs/consistency_model/unet_eval.yaml
```

### BS-RoFormer Experiments (MUSDB18)

Code coming soon!

---

## Acknowledgments

This codebase builds upon and integrates ideas and components from the following repositories:

- [Sony CTM](https://github.com/sony/ctm)
- [Multi-Source Diffusion Models](https://github.com/gladia-research-group/multi-source-diffusion-models)
- [Audio Diffusion PyTorch (Version 0.43)](https://github.com/archinetai/audio-diffusion-pytorch)
- [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (BS-RoFormer backbone)

We greatly appreciate the authors of these repositories for their contributions to the field and for making their work publicly available.

---

## Citations

```bibtex
@misc{karchkhadze2024improvingsourceextractiondiffusion,
  title={Improving Music Source Separation with Diffusion and Consistency Refinement},
  author={Tornike Karchkhadze and Mohammad Rasool Izadi and Shuo Zhang and Shlomo Dubnov},
  year={2024},
  eprint={2412.06965},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2412.06965},
}
```
