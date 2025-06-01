# PhishClip 🛡️
A CLIP-based phishing site screenshot classifier with CoOp (Context Optimization).

This project implements a vision-language model that detects phishing websites from screenshots using soft prompt tuning. It compares performance against baseline models such as Phishpedia and zero-shot CLIP.

---

## 📌 Project Overview

Phishing sites visually mimic trusted login pages to trick users into giving away credentials. Traditional defenses like URL blacklists struggle with these attacks.

**PhishClip** leverages the power of:
- **CLIP** (Contrastive Language-Image Pretraining)
- **CoOp** (Soft Prompt Tuning)

to enhance screenshot-based phishing detection in a few-shot setting.

---

## 📂 Repository Structure

```
.
├── CoOp_train.py          # Main training script for CoOp
├── shot.png               # Sample screenshot from dataset
├── train_dataset.json     # Sample dataset entry
├── report.tex             # LaTeX final report
├── accuracy_by_model.png  # Accuracy plot
├── confidence_by_model.png
├── confusion_matrix_*.png # Confusion matrices for models
├── README.md              # This file
└── .gitignore             # Ignore rules (including dataset/)
```

---

## 🛠️ Setup Instructions

### 1. Environment

```bash
# Recommended: use virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn tqdm
```

> ⚠️ Requires:  
> - Python 3.10.12  
> - PyTorch 2.7.0 + CUDA 11.8  
> - NVIDIA GPU (e.g., RTX 3060 Ti) for acceleration

---

## 🚀 Running the CoOp Trainer

```bash
python CoOp_train.py
```

Outputs include training logs, saved prompt parameters, and performance metrics (accuracy, F1-score, confidence, etc.).

---

## 📊 Dataset (Not Included)

Due to size constraints, the `dataset/` folder is **excluded** from version control.

You can download the **Phishpedia Benchmark Dataset** from:

🔗 https://sites.google.com/view/phishpedia-site/

After downloading, place it under:

```
project_root/
└── dataset/
    └── benign_sample_30k/
        └── brand/
            └── shot.png
```

---

## 🧾 Reproducibility Notes

- Training is done over 5 epochs using 100% of the data (reshuffled each epoch).
- Evaluation is on a 20% held-out test split.
- Prompt parameters are randomly initialized as `(C, L, D)` tensors.
- CoOp uses cross-entropy loss and the Adam optimizer (`lr=1e-3`).

---

## 📄 Report

A full LaTeX report is available in `report.tex`, including:
- Introduction and motivation
- Methodology and training details
- Results (accuracy/F1, confusion matrix)
- Appendix with code, dataset sample, and screenshots

---

## 📌 Citation & References

- Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP)*. ICML.
- Zhou et al. (2022). *Learning to Prompt for Vision-Language Models (CoOp)*. CVPR.
- Zhang et al. (2021). *Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages*. USENIX Security.

---

## 🌍 UN SDG Contribution

This project supports **UN Sustainable Development Goal #10 (Reduced Inequalities)** by improving phishing detection for digitally underserved users, reducing risks from cybercrime and disinformation.

---

## 🔒 License

This project is for academic and research purposes only. Contact the author for reuse beyond that.
