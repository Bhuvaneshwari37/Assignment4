 🐟 Fish Image Classification with CNN & Pretrained Models
This repository contains a deep learning project for **classifying fish images into multiple species** using a **custom Convolutional Neural Network (CNN)** and several **pretrained models** including **VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0**.
The project covers the complete pipeline: **data preprocessing, model training, evaluation, deployment, and performance comparison**.

📑 Table of Contents

- Project Overview  
- Installation  
- Imports  
- Dataset  
- Directory Structure  
- Preprocessing  
- Models  
- Training  
- Evaluation  
- Results  
- Usage  
- Class Names  
- Visualizations  
- License  
📌 Project Overview
The objective of this project is to **classify fish species using deep learning** and compare the performance of a **custom CNN** against **state-of-the-art pretrained models**.

Key Highlights
**Dataset:** Images labeled into **11 fish categories**
 **Techniques:**  
  - Custom CNN  
  - Transfer Learning & Fine-tuning  
  - Data Augmentation  
 **Models Compared:**  
  - VGG16  
  - ResNet50  
  - MobileNet  
  - InceptionV3  
  - EfficientNetB0  
**Evaluation Metrics:**  
  Accuracy, Precision, Recall, F1-Score, Confusion Matrix

⚙️ Installation
Install all required dependencies:
pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn tqdm
pip install streamlit pyngrok streamlit-folium nbconvert
npm install localtunnel
📦 Imports
python
Copy code
# File Handling and Image Processing
import os
import zipfile
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
# Torch and Vision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Display and Progress
from IPython.display import display
from tqdm import tqdm

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Warnings
import warnings
warnings.filterwarnings('ignore')
🗂 Dataset
The dataset consists of labeled fish images, divided into:

train/ – Training data

val/ – Validation data

test/ – Testing data

Each folder contains subdirectories representing individual fish classes.

📁 Directory Structure
bash
Copy code
Dataset/
├── train/
│   ├── class1/
│   ├── class2/
├── val/
│   ├── class1/
│   ├── class2/
├── test/
│   ├── class1/
│   ├── class2/
🔄 Preprocessing
The following preprocessing techniques are applied:

Resize images to 224 × 224

Random horizontal flipping

Random rotation (-15° to +15°)

Random affine transformations

Normalization using ImageNet mean and standard deviation

🧠 Models
🔹 Custom CNN
2 Convolutional layers
•	ReLU activation
•	MaxPooling
•	Dropout layers
•	Fully connected layers for classification

🔹 Pretrained Models (Transfer Learning)
•	VGG16
•	ResNet50
•	MobileNet
•	InceptionV3
•	EfficientNetB0

Final classification layers were modified for 11 classes, for example:
python
Copy code
vgg16.classifier[6] = nn.Linear(num_features, 11)
⚙️ Loss Function & Optimizer
Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 0.001

🏋️ Training
Epochs: 20
Batch Size: 32
Training Loader: train_loader
Validation Loader: val_loader
Data shuffling enabled during training

Model checkpointing to save the best .pth model
Each pretrained model was trained and validated using the same pipeline for fair comparison.
📊 Evaluation
Evaluation was performed on the test dataset using:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix

🏆 Results
🔍 Model Performance Comparison
Model	Accuracy	Precision	Recall	F1 Score
VGG16	85.5%	84.0%	85.0%	84.5%
ResNet50	87.2%	86.5%	87.0%	86.7%
MobileNet	89.0%	88.0%	89.2%	88.6%
InceptionV3	90.3%	89.5%	90.0%	89.7%
EfficientNetB0	91.1%	90.5%	91.2%	90.9%

✔ EfficientNetB0 achieved the best overall performance.

🚀 Usage
Run the Streamlit application locally or in Google Colab:

bash
Copy code
streamlit run /content/app.py &>/content/logs.txt & npx localtunnel --port 8501
🐠 Class Names
The 11 fish species used for classification are:
•	animal fish
•	animal fish bass
•	fish sea_food black_sea_sprat
•	fish sea_food gilt_head_bream
•	fish sea_food hourse_mackerel
•	fish sea_food red_mullet
•	fish sea_food red_sea_bream
•	fish sea_food sea_bass
•	fish sea_food shrimp
•	fish sea_food striped_red_mullet
•	fish sea_food trout

📈 Visualizations
Training & validation accuracy/loss curves
Confusion matrices
Sample predictions with ground truth labels
Model-wise performance comparison plots
