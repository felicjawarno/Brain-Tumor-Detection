# Brain Tumor Detection

This project implements a deep learning pipeline for detecting brain tumors in MRI images using transfer learning with a pre-trained VGG19 model.

## Overview

- **Task**: Binary classification (Tumor vs No Tumor)
- **Model**: VGG19 (pre-trained on ImageNet)
- **Dataset**: Brain MRI dataset labeled as "YES" (tumor) or "NO" (no tumor)
- **Preprocessing**:
  - Automatic cropping of brain areas using contour detection
  - Resizing images to 224x224 pixels
- **Training**:
  - Data augmentation to reduce overfitting
  - Class balancing with computed class weights
  - Early stopping and learning rate reduction callbacks

## Pipeline

1. **Preprocessing**:
    - Automatic brain area cropping and resizing
    - Stratified split into Train (70%), Validation (15%), and Test (15%) sets

2. **Model Architecture**:
    - Base: VGG19 without top layers (frozen)
    - Custom head:
      - Flatten
      - Dense layers with Batch Normalization and Dropout
      - Output: Single neuron with sigmoid activation

3. **Training**:
    - Optimizer: Adam with learning rate scheduling
    - Data augmentation on training data
    - Early stopping on validation loss

4. **Evaluation**:
    - Visualization of accuracy and loss curves
    - Final evaluation on the test set

## Results

- Achieved solid performance on the test set
- Included plots for training and validation metrics

## Dependencies

```bash
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
tqdm
