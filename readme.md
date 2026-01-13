# CIFAR-10 Image Classification using PyTorch

This project implements a **Convolutional Neural Network (CNN)** in PyTorch to classify images from the **CIFAR-10 dataset**. It includes training, evaluation, and visualization of model performance, including sample predictions and a confusion matrix.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results & Visualization](#results--visualization)  
- [License](#license)  

---

## Project Overview

This project trains a small CNN to classify images from the CIFAR-10 dataset, which contains **60,000 color images** in 10 classes. The pipeline includes:  

- Data loading with augmentation for training  
- CNN model definition with batch normalization and dropout  
- Training with **Adam optimizer** and **cross-entropy loss**  
- Evaluation with accuracy metrics, classification report, and confusion matrix  
- Visualization of sample predictions  

---

## Dataset

**CIFAR-10** dataset consists of:

- 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`  
- 50,000 training images and 10,000 test images  
- Each image is **32x32 pixels** with 3 color channels  

The dataset is automatically downloaded when running the code.

---

## Model Architecture

The `SimpleCNN` model consists of:

1. **Feature Extractor:**  
   - 3 convolutional layers (`Conv2d`) with ReLU activations  
   - Batch normalization after each convolution  
   - Max pooling layers to reduce spatial dimensions  

2. **Classifier:**  
   - Fully connected layer with 256 units + ReLU  
   - Dropout layer (0.3) for regularization  
   - Final linear layer outputting 10 class scores  

---

## Installation

### Requirements

- Python 3.8+  
- PyTorch 2.0+  
- torchvision  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

Install dependencies using:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
