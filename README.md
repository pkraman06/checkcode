# 🧠 Brain Tumor MRI Classification with Explainability & Training Dynamics

An end-to-end deep learning framework for **multi-class brain tumor classification** from MRI scans using a custom **Residual CNN with Squeeze-and-Excitation (SE) Attention**. In addition to accurate classification, this project provides **model explainability** through **Grad-CAM** and **Saliency Maps**, along with **representation learning analysis** using **Linear Probing** across network depth and training epochs.

---

# 📌 Executive Summary

Brain tumors are among the most aggressive forms of cancer, making **early and accurate diagnosis** essential for treatment planning and improving patient outcomes.

Magnetic Resonance Imaging (**MRI**) is the primary imaging modality for detecting brain tumors. However, manual interpretation can be time-consuming and may vary between radiologists, especially when tumors exhibit similar visual characteristics.

This project focuses on three major goals:

- **High Classification Accuracy**
  - Custom Residual CNN with **Squeeze-and-Excitation (SE) Attention**
  - Classifies MRI scans into four categories:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - No Tumor

- **Model Explainability**
  - **Grad-CAM** for localization of discriminative regions
  - **Vanilla Gradient Saliency Maps** for pixel-level importance visualization

- **Representation Learning Analysis**
  - Linear probing across intermediate feature layers
  - Temporal probing across training checkpoints
  - Quantifies how learned representations improve during training

---

# 🎯 Classification Categories

| Class | Description | MRI Characteristics |
|--------|-------------|--------------------|
| **Glioma** | Tumor originating from glial cells | Irregular borders, heterogeneous intensity, surrounding edema |
| **Meningioma** | Tumor arising from the meninges | Well-defined extra-axial mass |
| **Pituitary** | Tumor located in the pituitary gland | Sellar or suprasellar mass |
| **No Tumor** | Healthy MRI scan | Normal anatomical structures |

---

# 🏗️ System Architecture

```text
Input MRI (224 × 224 × 3)
           │
           ▼
Stem Convolution
           │
           ▼
Stage 1 (Residual + SE)
           │
           ├── Linear Probe 1
           ▼
Stage 2 (Residual + SE)
           │
           ├── Linear Probe 2
           ▼
Stage 3 (Residual + SE)
           │
           ├── Linear Probe 3
           ▼
Stage 4 (Residual + SE)
           │
           ├── Linear Probe 4
           ├── Grad-CAM Target Layer
           ▼
Global Average Pooling
           │
           ▼
Fully Connected Layer
           │
           ▼
Prediction
```

---

# 🚀 Features

- ✅ Custom Residual CNN Architecture
- ✅ Squeeze-and-Excitation (SE) Attention
- ✅ Mixed Precision (AMP) Training
- ✅ AdamW Optimizer
- ✅ Cosine Annealing Warm Restarts Scheduler
- ✅ Label Smoothing
- ✅ Grad-CAM Visualization
- ✅ Saliency Maps
- ✅ Linear Probing Across Network Layers
- ✅ Temporal Representation Analysis
- ✅ Interactive Gradio Interface

---

# 🧠 Model Architecture

The model is built using a custom **Residual Convolutional Neural Network** enhanced with **Squeeze-and-Excitation (SE) blocks**.

### Residual Connections

Residual blocks improve optimization by enabling gradient flow through skip connections, making deeper architectures easier to train.

### Squeeze-and-Excitation Attention

SE Attention recalibrates channel-wise feature responses by:

1. Aggregating spatial information
2. Learning channel importance
3. Emphasizing informative tumor features
4. Suppressing irrelevant background information

---

# ⚙️ Training Strategy

### Optimizer

- AdamW

### Learning Rate Scheduler

- Cosine Annealing with Warm Restarts

### Mixed Precision

- PyTorch Automatic Mixed Precision (AMP)

### Regularization

- Label Smoothing
- Weight Decay

---

# 🔬 Representation Learning Analysis

Unlike conventional classification repositories, this project analyzes **how the network learns**.

## 1. Layer-wise Linear Probing

Intermediate features from each residual stage are frozen and evaluated using independent linear classifiers.

This measures:

- Feature quality
- Linear separability
- Information progression through the network

---

## 2. Temporal Probing

Embeddings from saved training checkpoints are analyzed to observe how class representations evolve throughout training.

This provides insights into:

- Representation maturity
- Learning dynamics
- Feature evolution over epochs

---

# 🔍 Explainability

## Grad-CAM

Grad-CAM highlights spatial regions that most influence the prediction by utilizing gradients flowing into the final convolutional layer.

Applications:

- Tumor localization
- Clinical interpretability
- Prediction verification

---

## Saliency Maps

Vanilla Gradient Saliency computes gradients of the prediction with respect to input pixels.

Provides:

- Pixel importance visualization
- Fine-grained sensitivity analysis
- Additional interpretability

---

# 📂 Project Structure

```text
Brain-Tumor-MRI-Classification/
│
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading and augmentations
├── download_data.py       # Kaggle dataset downloader
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation and metrics
├── model.py               # Residual CNN + SE Attention
├── explainability.py      # Grad-CAM & Saliency Maps
├── probe.py               # Linear probing utilities
├── app.py                 # Gradio interface
├── utils.py               # Utility functions
│
├── checkpoints/
├── results/
├── data/
└── README.md
```

---

# 📦 Dataset

The project uses the **Brain Tumor MRI Dataset** from Kaggle.

Dataset structure:

```text
data/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Download automatically:

```bash
python download_data.py
```

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Brain-Tumor-MRI-Classification.git

cd Brain-Tumor-MRI-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Train

```bash
python train.py
```

Training generates:

- Model checkpoints
- Training history
- Accuracy curves
- Validation metrics

---

## Evaluate

```bash
python evaluate.py
```

Outputs:

- Test Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Representation Probing

```bash
python probe.py
```

Generates:

- Layer-wise probing results
- Checkpoint probing results
- Representation analysis plots

---

## Launch Gradio App

```bash
python app.py
```

The interface includes:

- MRI Upload
- Prediction
- Confidence Scores
- Grad-CAM Visualization
- Saliency Maps
- Representation Analysis

---

# 📊 Generated Outputs

The `results/` directory contains:

```text
results/
│
├── training_curves.png
├── history.json
├── confusion_matrix.png
├── classification_report.json
├── probe_by_layer.png
├── probe_by_checkpoint.png
└── explainability_examples/
```

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 🧰 Technologies Used

## Deep Learning

- PyTorch
- Torchvision

## Machine Learning

- Scikit-learn
- NumPy
- Pandas

## Image Processing

- OpenCV
- Pillow

## Visualization

- Matplotlib
- Seaborn

## Explainability

- Grad-CAM
- Saliency Maps

## Web Interface

- Gradio

---

# 🔮 Future Improvements

- Vision Transformers (ViT)
- EfficientNet backbone
- MONAI integration
- SHAP explanations
- LIME explanations
- Cross-validation experiments
- Test-Time Augmentation
- ONNX/TorchScript deployment
- Docker support
- Model quantization

---

# 📚 References

- Brain Tumor MRI Dataset (Kaggle)
- Grad-CAM: Visual Explanations from Deep Networks
- Squeeze-and-Excitation Networks (CVPR 2018)
- Deep Residual Learning for Image Recognition (ResNet)

---

# 📄 License

This project is released under the **MIT License**.

---

# ⭐ Acknowledgements

- PyTorch
- Scikit-learn
- OpenCV
- Gradio
- Kaggle Brain Tumor MRI Dataset
