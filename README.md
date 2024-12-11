# Improving Source Extraction with Diffusion and Consistency Models

<p align="center"></p>

This repository houses the official PyTorch implementation of the paper titled **"Improving Source Extraction with Diffusion and Consistency Models"** on the Slakh2100 dataset. The paper was presented as an oral presentation at **NeurIPS 2024 Workshop Audio Imagination: AI-Driven Speech, Music, and Sound Generation**.

- [arXiv](https://arxiv.org/abs/2412.06965)
- [Demo](https://consistency-separation.github.io/)
- [OpenReview](https://openreview.net/forum?id=nskR7tWE6z)

**Contacts**:
- Tornike Karchkhadze: [tkarchkhadze@ucsd.edu](mailto:tkarchkhadze@ucsd.edu)  
- Mohammad Rasool Izadi: [russell_izadi@bose.com](mailto:russell_izadi@bose.com)

*The work was done during Tornike's internship at Bose Corporation.

---

## Abstract

In this work, we integrate a score-matching diffusion model into a standard deterministic architecture for time-domain musical source extraction. To address the typically slow iterative sampling process of diffusion models, we apply consistency distillation and reduce the sampling process to a single step, achieving performance comparable to that of diffusion models. With two or more steps, the model even surpasses diffusion models. Trained on the Slakh2100 dataset for four instruments (bass, drums, guitar, and piano), our model demonstrates significant improvements across objective metrics compared to baseline methods.

---

## Checkpoints

Please contact the authors for checkpoints. 

---

## Prerequisites

### 1. Dataset

In this project, the Slakh2100 dataset is used.  
Please follow the instructions for data download and setup provided here:  
[Slakh2100 Data Setup](https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md)

### 2. Conda Environment Setup

This repository uses Python 3.9.19.  

```bash
# Create environment
conda env create -f environment.yaml

# Activate environment
conda activate ctm
```

---

## Training

### Deterministic Model Training
```bash
python train_audio_simple.py --cfg configs/deterministic_model/cond_separation_simple_no_diff_train.yaml
```

### Diffusion Model Training
```bash
python train_audio.py --cfg configs/diffusion_model/train_audiodm_cond_separation_unet_every_layer_pre_trained_feature_extractor.yaml
```

### Consistency Model Training
```bash
python main_audio_ctm.py --cfg configs/consistency_model/CD_sourse_extraction_unet_every_layer_pre_trained_feature_extractor_train.yaml
```

---

## Sampling and Evaluation

### Deterministic Model Evaluation
```bash
python train_audio_simple.py --cfg configs/deterministic_model/cond_separation_simple_no_diff_eval.yaml
```

### Diffusion Model Evaluation
```bash
python train_audio.py --cfg configs/diffusion_model/Diff_cond_separation_unet_every_layer_pre_trained_feature_extractor_eval_MSDMSampler.yaml
```

### Consistency Model Evaluation
```bash
python main_audio_ctm.py --cfg configs/consistency_model/CD_sourse_extraction_unet_every_layer_pre_trained_feature_extractor_eval.yaml
```
---

## Acknowledgments

This codebase builds upon and integrates ideas and components from the following repositories:

- [Sony CTM](https://github.com/sony/ctm)  
- [Multi-Source Diffusion Models](https://github.com/gladia-research-group/multi-source-diffusion-models)  
- [Audio Diffusion PyTorch (Version 0.43)](https://github.com/archinetai/audio-diffusion-pytorch)  

We greatly appreciate the authors of these repositories for their contributions to the field and for making their work publicly available.

--- 

## Citations

```bibtex

@misc{karchkhadze2024improvingsourceextractiondiffusion,
  title={Improving Source Extraction with Diffusion and Consistency Models}, 
  author={Tornike Karchkhadze and Mohammad Rasool Izadi and Shuo Zhang},
  year={2024},
  eprint={2412.06965},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2412.06965}, 
}
@inproceedings{
  anonymous2024improving,
  title={Improving Source Extraction with Diffusion and Consistency Models},
  author={Anonymous},
  booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=nR2DHRxWS2},
  note={under review}
}
```
