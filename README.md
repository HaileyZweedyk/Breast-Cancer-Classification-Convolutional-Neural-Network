# Breast Cancer Ultrasound Classification (CNN)

## Technologies Used

Python (tensorflow, scikit-learn, matplotlib, pandas, numpy, os, kagglehub)

## Overview
This project uses a Convolutional Neural Network (CNN) with transfer learning to classify breast ultrasound images into three categories:

- **Benign**
- **Malignant**
- **Normal**

The goal is to support medical image analysis by improving detection consistency and reducing missed malignant cases.

Breast cancer accounts for ~30% of new cancer diagnoses among women in the U.S., with ~321,910 cases annually and a lifetime risk of ~13%. While radiologists are highly accurate, machine learning models can assist by identifying subtle patterns and improving consistency in diagnosis.

---

## Dataset
- **Source:** Kaggle Breast Ultrasound Images Dataset
- **Size:** 1,578 images  
- **Classes:** Benign, Malignant, Normal  
- **Challenge:** Class imbalance across categories  

---

## Objective
Due to the limited dataset size, the model aims to achieve:

- ≥85% **Malignant Recall**

### Primary Focus
Maximize **recall for malignant tumors** to minimize false negatives, which is critical in medical diagnosis.

---

## Model Architecture

- **Base Model:** ResNet50 (transfer learning)
- **Input Size:** 200 × 200  
- **Feature Extraction:** Global Average Pooling → 2048-dimensional vector  

### Custom Layers
- Dense Layer: 128 neurons (ReLU)
- Dropout: 30%
- Output Layer: Softmax (3 classes)

---

## Training Strategy

The model was trained in three stages:

### 1. Frozen Base Model
- Trainable: `False`  
- Optimizer: Adam  
- Epochs: 10  

### 2. Full Fine-Tuning
- Trainable: `True`  
- Optimizer: Adam (1e-5)  
- Epochs: 10  

### 3. Partial Fine-Tuning
- Trainable: Last 40 layers  
- Optimizer: Adam (1e-5)  
- Epochs: 10  

---

## Handling Class Imbalance

Initial results showed:
- High accuracy and precision  
- **Low recall (<80%)**, especially malignant (<60%)

### Solution: Class Weights
Adjusted weights to prioritize malignant detection:

| Class      | Weight |
|------------|--------|
| Normal     | 1.2    |
| Malignant  | 2.3    |
| Benign     | 1.0    |

Increasing malignant weight to 2.5 resulted in performance decline, confirming **2.3 as optimal**.

---

## Results

| Metric            | Score |
|------------------|------|
| Accuracy         | 91%  |
| Malignant Recall | 90%  |
| Benign Recall    | 94%  |
| Normal Recall    | 79%  |

### Key Insight
- Strong malignant detection (primary objective achieved)
- Lower performance on "normal" class due to class imbalance and weighting trade-offs

---

## Key Takeaways
- Class weighting is critical for imbalanced medical datasets  
- Optimizing recall for high-risk classes improves real-world usefulness  
- Smaller datasets can lead to early overfitting, making shorter training more effective  

---

## Future Improvements
- Increase dataset size  
- Apply data augmentation techniques  
- Further hyperparameter tuning  
- Experiment with alternative architectures or ensembles  
 

---

## Disclaimer
This project is for research and educational purposes only and is not intended for clinical use.
