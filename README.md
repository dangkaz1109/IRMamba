# IR-MambaAD: Infrared Multi-class Unsupervised Anomaly Detection

> A reconstruction-based framework for **multi-class unsupervised anomaly detection on infrared imagery**, combining Multi-Scale Wavelet Feature Modulation, a DINO-pretrained ResNet-34 encoder, and a High-Frequency Prior Guided Mamba Decoder.

---

## 📌 Overview

**IR-MambaAD** addresses the problem of **multi-class unsupervised industrial anomaly detection (UIAD)** on infrared (IR) images. Unlike RGB-based methods, infrared images exhibit distinct spectral and textural properties that challenge standard feature extraction pipelines. IR-MambaAD tackles this by:

- **Enhancing high-frequency (HF) cues** in IR images before they reach the encoder, using a Multi-Scale Wavelet Feature Modulation (MWFM) module.
- **Bridging the RGB–IR domain gap** via DINO-based self-supervised pretraining on unlabeled IR data.
- **Reconstructing normal features** with a High-Frequency Prior Guided Mamba Decoder (HPG-Mamba) that conditions state-space updates on wavelet-derived HF priors.
- **Detecting anomalies** at inference time from reconstruction errors between input and reconstructed feature maps.


## 🚀 Quick Start

## 📋 Requirements
```bash
pip install -r requirements.txt
```


### Option 1 — Using Pretrained Models

1. Download the **main pretrained model**: [Download here](https://drive.google.com/file/d/1hpt-t7WOJ8QwU-1_zrRG2zgTIL7dLJFG/view?usp=sharing) → place in `data/`
2. Download the **DINO pretrained backbone**: [Download here](https://drive.google.com/file/d/1aBPftvO5Metttw0XaHrcriVA9UTsQ6qn/view?usp=sharing) → place in `data/dino_pretrain/`
3. Run evaluation:

```bash
python evaluate.py
```

### Option 2 — Training From Scratch

Training follows a two-stage pipeline:

**Stage 1 — DINO Self-Supervised Pretraining** (trains MWFM + encoder, decoder frozen):

```bash
python train_dino.py
```

**Stage 2 — Decoder Training** (MWFM + encoder frozen, trains H-FPN + HPG-Mamba decoder):

```bash
python train_decoder.py
```

---

## 📊 Evaluation

```bash
python evaluate.py
```


## 📁 Project Structure

```
IRMamba/
├── data/                    # Dataset and pretrained weights
│   └── dino_pretrain/       # DINO pretrained backbone
├── models/                  # Model architecture definitions
│   └── ...                  # MWFM, HPG-Mamba, H-FPN, etc.
├── dataset.py               # Data loading and preprocessing
├── train_dino.py            # Stage 1: DINO self-supervised pretraining
├── train_decoder.py         # Stage 2: HPG-Mamba decoder training
├── evaluate.py              # Evaluation and anomaly map generation
├── requirements.txt
└── README.md
```

---

## 📖 Citation

If you use this work, please cite the original paper and MambaAD which this builds upon:

```bibtex
@article{he2024mambaad,
  title   = {MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection},
  author  = {He, Haoyang and Bai, Yuhu and Zhang, Jiangning and others},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {37},
  pages   = {71162--71187},
  year    = {2024}
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
