# Human Activity Recognition using Deep Learning

## Overview

This project explores **Human Activity Recognition (HAR)** using deep learning on time-series sensor data.

The goal is to understand how different deep learning approaches perform on sequential data, starting from simple supervised models and gradually moving towards more advanced techniques like **self-supervised learning and transfer learning**.

---

## Motivation

Most real-world datasets do not have large amounts of labeled data. Labeling is expensive and time-consuming.

This project investigates:

* How well models perform with **limited labeled data**
* Whether **self-supervised learning (SSL)** can help improve performance
* How different architectures behave on time-series data

---

## Dataset

* **KU-HAR (Korea University Human Activity Recognition)**
* Shape: `(20750, 6, 300)`
  * 6 channels (sensor signals)
  * 300 time steps per sample
* 18 activity classes (walking, sitting, running, etc.)
* Source: https://data.mendeley.com/datasets/45f952y38r/5

---

## Project Structure

```
project/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   └── load.py     # data loading & preprocessing
│
├── notebooks/
|   ├── 01_data_preprocessing.ipynb
│   └── 02_baseline_cnn.ipynb
│
└── README.md
```

---

## Approach

### 1. Baseline Model

* 1D CNN for time-series classification
* Fully supervised learning

### 2. Self-Supervised Learning

* Autoencoder to reconstruct input signals
* Masked Autoencoder for improved learning

### 3. Transfer Learning

* Reuse pretrained encoder for classification
* Experiments with:
  * Full labeled data
  * Limited labeled data (20%)

---

## Goals

* Compare **baseline vs self-supervised models**
* Study performance under **limited labels**
* Improve representation learning using **masked inputs**

---