# Iris Recognition using Deep Learning (Classification Approach)

## Overview

This project presents an iris recognition system formulated as a multi-class classification problem. Each individual is treated as a separate class, and a convolutional neural network (CNN) is trained to predict identity directly from iris images.

The implementation includes a structured codebase (`src/`) along with a notebook demonstrating the complete pipeline from data preprocessing to model training and evaluation.

---

## Motivation

Traditional iris recognition systems rely on handcrafted feature extraction and template matching for verification. While effective, these approaches can be complex and less scalable.

This project explores a classification-based alternative, where a deep learning model learns discriminative iris features in an end-to-end manner. This simplifies the pipeline and enables efficient inference.

---

## Methodology

### Data Preparation

* Images are resized to a uniform resolution
* Normalization is applied
* Data is organized into class-labeled directories (one class per identity)
* Train-test split is performed

### Model

* Convolutional Neural Network (CNN)
* Convolution and pooling layers for feature extraction
* Fully connected layers for classification
* Softmax output for class probabilities

### Training

* Loss Function: Cross-Entropy Loss
* Optimizer: Adam
* Mini-batch training over multiple epochs

### Evaluation

* Model performance is assessed on a held-out test set
* Confusion matrix and prediction outputs are visualized in the notebook

---

## Project Structure

```id="w8j3s2"
iris-recognition/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Usage

### Run the notebook

```bash id="m1z8k4"
jupyter notebook notebook.ipynb
```

### (Optional) Run scripts

```bash id="q7n2p6"
python src/train.py
python src/eval.py
```

---

## Dataset

This project uses the **CASIA Iris-Thousand (CASIA-Iris-1000)** dataset.

The dataset is not included in this repository. Please download it separately and organize it as follows:

```id="z5x1c8"
data/
 ├── class_1/
 │    ├── img1.jpg
 │    ├── img2.jpg
 │
 ├── class_2/
 │    ├── img1.jpg
 │    └── ...
```

Each class corresponds to a unique identity.

---


## Authors

* Anirudh Sairam
* Anurag Thakur
* Aravind Sarath Chandran
* Kanak Potdar
* Nikhil N
