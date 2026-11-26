"""
CNN Training Script for Diabetic Retinopathy Classification
RetinaScan AI - Deep Learning Pipeline
Architecture: 4 Conv Blocks, GAP, Dense(256), Softmax(5 Grades)

Architecture specified in Slide 6 & Slide 9 of PPT:
- Input: 224x224x3 retinal image
- Convolution layers with ReLU activation
- Pooling layers (Down-sampling)
- Feature map extraction
- Dense classification layer
- Softmax output with 5 classes (Grades 0-4)
- Saves model to backend/model/dr_model.keras
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from config import (
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SAVED_MODEL_KERAS,
    EVALUATION_METRICS_PATH
)
from dataset_loader import (
    load_dataset_metadata,
    split_dataset,
    create_tf_dataset
)

def build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Constructs the CNN architecture following the project presentation:
    Input (224x224x3) -> Conv + ReLU -> Pool -> Conv + ReLU -> Pool ->
    Conv + ReLU -> Pool -> Flatten / Dense -> Softmax (5 grades).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="retinal_input"),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_32"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_64"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_128"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),

        # Conv Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_256"),
        layers.BatchNormalization(name="bn4"),
        layers.MaxPooling2D((2, 2), name="pool4"),

        # Feature Abstraction & Classifier
        layers.GlobalAveragePooling2D(name="global_pool"),
        layers.Dense(256, activation="relu", name="dense_features"),
        layers.Dropout(0.4, name="dropout_regularization"),
        layers.Dense(num_classes, activation="softmax", name="softmax_5_grades")
    ], name="Diabetic_Retinopathy_CNN")

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("=" * 70)
    print("RETINASCAN AI - CNN MODEL TRAINING PIPELINE")
    print("Clinical Decision Support System (CDSS)")
    print("=" * 70)

    # 1. Discover dataset
    image_paths, labels = load_dataset_metadata()
    if len(image_paths) == 0:
        print("\n[Notice] No dataset images found in 'dataset/' directory.")
        print("To train on APTOS 2019 or EyePACS:")
        print("  1. Download dataset (e.g., from Kaggle: aptos2019-blindness-detection)")
        print("  2. Place 'train.csv' in dataset/ and images in dataset/train_images/")
        print("  3. Re-run: python training/train.py\n")
        print("Building and validating CNN model architecture definition...")
        model = build_cnn_model()
        model.summary()
        return

    print(f"\n[Dataset] Total valid labelled images found: {len(image_paths)}")
    print(f"[Dataset] Class counts: { {c: labels.count(c) for c in range(NUM_CLASSES)} }")

    # 2. Train / Val / Test Splits
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = split_dataset(image_paths, labels)
    print(f"[Split] Train: {len(train_p)}, Validation: {len(val_p)}, Test: {len(test_p)}")

    # 3. Create Datasets
    train_ds = create_tf_dataset(train_p, train_y, batch_size=BATCH_SIZE, is_training=True)
    val_ds = create_tf_dataset(val_p, val_y, batch_size=BATCH_SIZE, is_training=False)
    test_ds = create_tf_dataset(test_p, test_y, batch_size=BATCH_SIZE, is_training=False)

    # 4. Build Model
    model = build_cnn_model()
    model.summary()

    # 5. Callbacks
    os.makedirs(os.path.dirname(SAVED_MODEL_KERAS), exist_ok=True)
    chk_callback = callbacks.ModelCheckpoint(
        SAVED_MODEL_KERAS,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    es_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 6. Fit Model
    print("\n[Training] Starting CNN training for up to", EPOCHS, "epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[chk_callback, es_callback]
    )

    # 7. Evaluate on independent Test set
    print("\n[Evaluation] Evaluating on independent test split...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[Evaluation Result] Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Format training curves for UI dashboard
    history_records = []
    epochs_trained = len(history.history["accuracy"])
    for ep in range(epochs_trained):
        history_records.append({
            "epoch": ep + 1,
            "train_acc": round(float(history.history["accuracy"][ep]), 4),
            "val_acc": round(float(history.history["val_accuracy"][ep]), 4),
            "train_loss": round(float(history.history["loss"][ep]), 4),
            "val_loss": round(float(history.history["val_loss"][ep]), 4)
        })

    # Save metrics JSON
    metrics_payload = {
        "is_real_training_data": True,
        "model_architecture": "Custom Retinal CNN (4 Conv Blocks, GAP, Dense Softmax)",
        "accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs_trained": epochs_trained,
        "total_train_samples": len(train_p),
        "total_val_samples": len(val_p),
        "total_test_samples": len(test_p),
        "training_history": history_records,
        "saved_model_path": SAVED_MODEL_KERAS
    }

    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[Complete] Model weights saved to: {SAVED_MODEL_KERAS}")
    print(f"[Complete] Evaluation metrics saved to: {EVALUATION_METRICS_PATH}")

if __name__ == "__main__":
    main()
