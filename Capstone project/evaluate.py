"""
Model Evaluation and Confusion Matrix Generation
RetinaScan AI - Deep Learning Pipeline
Clinical Decision Support System (CDSS)

Computes metrics following Slide 13 of project presentation:
- Accuracy
- Precision
- Recall / Sensitivity
- F1-Score
- 5x5 Confusion Matrix
- Saves structured evaluation results to backend/model/evaluation_results.json
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)

from config import (
    SAVED_MODEL_KERAS,
    EVALUATION_METRICS_PATH,
    NUM_CLASSES,
    CLASS_NAMES
)
from dataset_loader import (
    load_dataset_metadata,
    split_dataset,
    preprocess_single_image
)

def evaluate_saved_model():
    print("=" * 70)
    print("DIABETIC RETINOPATHY MODEL EVALUATION")
    print("=" * 70)

    if not os.path.exists(SAVED_MODEL_KERAS):
        print(f"[Error] No trained model weights found at {SAVED_MODEL_KERAS}.")
        print("Please train the model first using 'python training/train.py'.")
        return

    print(f"[Loading] Loading model weights from {SAVED_MODEL_KERAS}...")
    model = tf.keras.models.load_model(SAVED_MODEL_KERAS)

    # Load dataset
    image_paths, labels = load_dataset_metadata()
    if len(image_paths) == 0:
        print("[Error] No dataset images available in dataset/ to evaluate.")
        return

    _, _, (test_paths, test_labels) = split_dataset(image_paths, labels)
    if len(test_paths) == 0:
        print("[Warning] Test set empty; falling back to evaluating on all images.")
        test_paths, test_labels = image_paths, labels

    print(f"[Evaluation] Processing {len(test_paths)} test images...")
    y_true = []
    y_pred = []

    for path, label in zip(test_paths, test_labels):
        try:
            img = preprocess_single_image(path)
            batch = np.expand_dims(img, axis=0)
            pred_probs = model.predict(batch, verbose=0)[0]
            pred_class = int(np.argmax(pred_probs))

            y_true.append(label)
            y_pred.append(pred_class)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class_summary = []
    for c in range(NUM_CLASSES):
        per_class_summary.append({
            "grade": c,
            "name": CLASS_NAMES[c],
            "precision": round(float(class_p[c]), 4),
            "recall": round(float(class_r[c]), 4),
            "f1_score": round(float(class_f1[c]), 4),
            "support": int(class_supp[c])
        })

    # Quadratic weighted kappa (standard ophthalmology metric)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        qwk = None

    print("\n" + "=" * 50)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {prec_macro * 100:.2f}%")
    print(f"Macro Recall:      {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    if qwk is not None:
        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print("=" * 50)
    print("Confusion Matrix (5x5):")
    print(cm)

    # Save to JSON
    results = {
        "is_real_training_data": True,
        "is_placeholder": False,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec_macro), 4),
        "recall": round(float(rec_macro), 4),
        "f1_score": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "quadratic_weighted_kappa": round(float(qwk), 4) if qwk is not None else None,
        "total_test_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_summary,
        "classes": CLASS_NAMES
    }

    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Evaluation metrics saved to {EVALUATION_METRICS_PATH}")

if __name__ == "__main__":
    evaluate_saved_model()
