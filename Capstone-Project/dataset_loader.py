"""
Dataset Loader and Preprocessing Pipeline for Retinal Images
Supports APTOS 2019 Blindness Detection, EyePACS, and custom folder layouts.
"""

import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import (
    DATASET_DIR,
    CSV_PATH,
    IMAGES_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)

def crop_dark_borders(img_bgr, tol=15):
    """Crops non-informative black margins around circular fundus."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return img_bgr
    row_sum = np.sum(mask, axis=1)
    col_sum = np.sum(mask, axis=0)
    rows = np.where(row_sum > 0)[0]
    cols = np.where(col_sum > 0)[0]
    return img_bgr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

def preprocess_single_image(image_path):
    """Loads and standardizes fundus image to (224, 224, 3) normalized [0, 1]."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    # Crop dark margins
    cropped = crop_dark_borders(img)
    # Resize to 224x224
    resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0.0, 1.0]
    norm = rgb.astype(np.float32) / 255.0
    return norm

def load_dataset_metadata():
    """
    Discovers image file paths and corresponding 0-4 labels.
    Supports:
    1. CSV file with id_code and diagnosis (APTOS format)
    2. Sub-folders named 0, 1, 2, 3, 4
    """
    image_paths = []
    labels = []

    # 1. Check for CSV format
    if os.path.exists(CSV_PATH) and os.path.exists(IMAGES_DIR):
        print(f"[Loader] Reading dataset metadata from {CSV_PATH}...")
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Find column names
            id_col = "id_code" if "id_code" in reader.fieldnames else ("image" if "image" in reader.fieldnames else reader.fieldnames[0])
            label_col = "diagnosis" if "diagnosis" in reader.fieldnames else ("level" if "level" in reader.fieldnames else reader.fieldnames[1])

            for row in reader:
                img_id = row[id_col].strip()
                label = int(row[label_col].strip())

                # Check extensions
                candidates = [
                    os.path.join(IMAGES_DIR, f"{img_id}.png"),
                    os.path.join(IMAGES_DIR, f"{img_id}.jpg"),
                    os.path.join(IMAGES_DIR, f"{img_id}.jpeg"),
                    os.path.join(IMAGES_DIR, img_id)
                ]
                found_path = None
                for c in candidates:
                    if os.path.exists(c):
                        found_path = c
                        break

                if found_path and 0 <= label < NUM_CLASSES:
                    image_paths.append(found_path)
                    labels.append(label)

    # 2. Check for sub-folder format (e.g. dataset/0, dataset/1 ...)
    elif os.path.exists(DATASET_DIR):
        for grade in range(NUM_CLASSES):
            grade_dir = os.path.join(DATASET_DIR, str(grade))
            if os.path.exists(grade_dir):
                for fname in os.listdir(grade_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(grade_dir, fname))
                        labels.append(grade)

    return image_paths, labels

def split_dataset(image_paths, labels):
    """Splits dataset into stratified Train, Validation, and Test sets."""
    if len(image_paths) == 0:
        return ([], []), ([], []), ([], [])

    # Train vs Temp (Val + Test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=temp_ratio, random_state=42, stratify=labels if len(set(labels)) > 1 else None
    )

    # Val vs Test
    val_rel_ratio = VAL_RATIO / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_rel_ratio), random_state=42, stratify=temp_labels if len(set(temp_labels)) > 1 else None
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_tf_dataset(paths, labels, batch_size=32, is_training=False):
    """Builds an optimized tf.data.Dataset pipeline with on-the-fly augmentation."""
    def generator():
        for p, l in zip(paths, labels):
            try:
                img = preprocess_single_image(p)
                yield img, tf.one_hot(l, NUM_CLASSES)
            except Exception as e:
                continue

    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if is_training:
        # Augmentations for training as specified in PPT Slide 8
        def augment(x, y):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, max_delta=0.1)
            x = tf.clip_by_value(x, 0.0, 1.0)
            return x, y

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=min(len(paths), 500))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
