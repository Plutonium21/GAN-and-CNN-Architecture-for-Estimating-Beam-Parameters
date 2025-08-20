
# GAN + CNN Architecture for Beam Parameter Estimation
### INDUS-2 Synchrotron Particle Accelerator | RRCAT, Indore

This repository contains a two-stage deep learning pipeline developed during my research internship at the **Raja Ramanna Centre for Advanced Technology (RRCAT)**. The project focuses on automating the estimation of beam parameters (position and shape) from Beam Position Monitor (BPM) images using Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview
In a synchrotron like **INDUS-2**, monitoring the electron beam's health is critical. Traditionally, this involves fitting Gaussian curves to imaging data, which can be computationally expensive or sensitive to noise. 

**This project implements:**
1.  **GAN-based Data Augmentation:** A Generative Adversarial Network trained to simulate high-fidelity beam profiles, mimicking the characteristics of real synchrotron imaging.
2.  **CNN Parameter Regressor:** A robust Convolutional Neural Network that predicts beam coordinates $(x, y)$ and beam sizes $(\sigma_x, \sigma_y)$ in real-time.

---

## 🏗️ Repository Structure

### 📂 CNN/ (Production Module)
The core regression pipeline for beam analysis.
* `model.py`: Defines the CNN architecture (Conv2D, BatchNormalization, GAP).
* `generator.py`: High-fidelity synthetic data generation with tilt, noise, and offsets.
* `train.py`: Script to train the regression model.
* `evaluate.py`: Performance metrics (RMSE, R²) and visualization scripts.
* `utils.py`: Image processing and physics-based Gaussian simulations.

### 📂 GAN/ (Research Module)
The generative pipeline used for data augmentation.
* `gan_training.py`: Implementation of a DCGAN to generate synthetic BPM images.
* `create_bpm_dataset.py`: Script to generate the initial training set for the GAN.
* `regressor.py`: A baseline model used to validate the quality of GAN-generated images.
* `evaluate_gan.py`: Evaluates the similarity between real and generated distributions.

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```
### 2. Running the CNN Pipeline
To generate the dataset and train the model:

```bash
python CNN/train.py
```
To evaluate the trained model and see visual predictions:
```bash
python CNN/evaluate.py
```

---
## 📊 Performance Results
The CNN architecture achieves high precision in estimating beam centers and widths, even in the presence of:

Rotational Tilt: Beam rotation between -45° and 45°.
Sensor Noise: Salt-and-pepper and Gaussian noise profiles.
Intensity Offsets: Robustness against variable background lighting.

Metrics:
-   **Position ($x, y$):**  $R^2$ scores of **0.92** and **0.89**.
    
-   **Overall RMSE:**  **1.77**

---
## 🔐 Note on Data
Due to the proprietary nature of the actual beam imaging data at RRCAT, the models in this repository are demonstrated using a High-Fidelity Synthetic Generator. This generator was specifically tuned to replicate the noise, tilt, and beam morphology observed in the INDUS-2 facility.

---
## 👨‍💻 Author
Pranshu Upadhyay, BTech CSE'28 | Shiv Nadar Institute of Eminence
Research Intern, Beam Diagnostics Section, RRCAT (2026)

---
## 🎓 Acknowledgments

Special thanks to Dr. Surendra Yadav (Scientific Officer-G, RRCAT) for technical guidance and the **Beam Diagnostics Section** for providing the operational context for the INDUS Accelerator Complex.