"""
Configuration for Diabetic Retinopathy CNN Training
RetinaScan AI - Deep Learning Pipeline
"""

import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset configuration
# Default expected folder structure:
# dataset/
#   ├── train.csv (id_code, diagnosis)
#   └── train_images/ (id_code.png)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")

# Image hyperparameters matching PPT Slide 8 & 9
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 5

CLASS_NAMES = [
    "0 - No DR",
    "1 - Mild NPDR",
    "2 - Moderate NPDR",
    "3 - Severe NPDR",
    "4 - Proliferative DR"
]

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Output model destinations
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
SAVED_MODEL_KERAS = os.path.join(SAVED_MODEL_DIR, "dr_model.keras")
EVALUATION_METRICS_PATH = os.path.join(SAVED_MODEL_DIR, "evaluation_results.json")
