# Flower Image Classifier

A deep learning image classifier that identifies 102 species of flowers using TensorFlow and transfer learning with MobileNetV2.

## Overview

This project trains a neural network on the [Oxford 102 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and wraps it into a command line application that can predict the species of a flower from any image.

- **Model**: MobileNetV2 (pretrained on ImageNet) + custom classifier head
- **Dataset**: Oxford 102 Flower Categories
- **Test Accuracy**: ~77%
- **Framework**: TensorFlow 2.x

## Project Structure

```
ImageClassifier/
├── notebook.ipynb          # Training notebook
├── notebook.html           # Exported notebook
├── predict.py              # Command line application
├── utils.py                # Image processing utilities
├── flower_classifier.h5    # Saved trained model
├── label_map.json          # Label to flower name mapping
└── test_images/
    ├── cautleya_spicata.jpg
    ├── hard-leaved_pocket_orchid.jpg
    ├── orange_dahlia.jpg
    └── wild_pansy.jpg
```

## Installation

Make sure you have Python 3.11 installed. It is recommended to use a conda environment:

```bash
conda create -n flower-classifier python=3.11
conda activate flower-classifier
conda install -c conda-forge pyarrow
pip install tensorflow tensorflow-datasets numpy matplotlib Pillow
```

## Usage

### Basic prediction

```bash
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg flower_classifier.h5
```

### Return top K most likely classes

```bash
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg flower_classifier.h5 --top_k 3
```

### Use a custom label map

```bash
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg flower_classifier.h5 --category_names label_map.json
```

### Example output

```
Prediction Results:
------------------------------
1. hard-leaved pocket orchid: 89.23%
2. canterbury bells: 4.12%
3. moon orchid: 2.01%
4. bird of paradise: 1.45%
5. windflower: 0.98%
```

## Model Architecture

The model uses transfer learning — MobileNetV2 pretrained on ImageNet is used as a feature extractor, with a custom classification head trained on the flower dataset.

```
MobileNetV2 (frozen, pretrained on ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dense(512, ReLU)
        ↓
Dropout(0.4)
        ↓
Dense(102, Softmax)
```

## Training

The model was trained for 20 epochs with the following configuration:

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Batch size**: 32
- **Image size**: 224×224
- **Data augmentation**: Random horizontal flips, random brightness

| Split      | Accuracy |
| ---------- | -------- |
| Training   | 99.8%    |
| Validation | 80.7%    |
| Test       | 77.6%    |

## Requirements

- Python 3.11
- TensorFlow 2.x
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow
