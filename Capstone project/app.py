"""
RetinaScan AI - Diabetic Retinopathy Detection & Analysis Platform
REST API Engine (Flask)

Modules:
- Module 1: Retinal Image Acquisition & Quality Inspection
- Module 2: OpenCV Image Preprocessing (Cropping, Resizing, Normalization, Augmentation)
- Module 3: CNN Classification & Model Performance Evaluation
"""

import os
import time
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from preprocessing.image_processor import (
    run_full_preprocessing_pipeline,
    check_image_quality,
    image_to_base64
)
from model.classifier import (
    predict_retinal_image,
    is_model_available,
    DR_GRADES
)
from model.metrics import get_model_evaluation_metrics

app = Flask(__name__)
# Enable CORS for all routes to seamlessly connect with React Vite frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "sample_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- PLATFORM METADATA -----------------
PROJECT_METADATA = {
    "platform_title": "RetinaScan AI - Diabetic Retinopathy Diagnostic Platform",
    "architecture": "Computer Vision & Deep Learning (CNN)",
    "edition": "Clinical Decision Support System (CDSS)",
    "standard": "ICDR 5-Grade Severity Classification",
    "modules": [
        {"id": 1, "title": "Retinal Image Acquisition", "description": "Color fundus photograph acquisition, quality validation and FOV inspection"},
        {"id": 2, "title": "Image Preprocessing", "description": "Dark border cropping, 224x224 resizing, pixel normalization and augmentation"},
        {"id": 3, "title": "CNN-Based Classification", "description": "Deep feature extraction and 5-grade DR severity classification"}
    ],
    "grades_reference": DR_GRADES
}

# ----------------- API ENDPOINTS -----------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Check backend service and environment availability."""
    model_loaded = is_model_available()
    return jsonify({
        "status": "healthy",
        "service": "RetinaScan AI Screening API",
        "edition": PROJECT_METADATA["edition"],
        "model_status": "weights_loaded" if model_loaded else "demonstration_mode",
        "is_model_available": model_loaded,
        "opencv_available": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }), 200

@app.route("/api/info", methods=["GET"])
def get_project_info():
    """Returns platform metadata and technical specifications."""
    return jsonify(PROJECT_METADATA), 200

@app.route("/api/samples", methods=["GET"])
def list_sample_images():
    """Returns list of curated retinal fundus samples for testing."""
    samples = []
    sample_files = [
        {"file": "grade0_normal_fundus.jpg", "grade": 0, "title": "Grade 0: Normal Fundus", "desc": "Clean optic disc, sharp vascular margins, no microaneurysms."},
        {"file": "grade1_mild_npdr.jpg", "grade": 1, "title": "Grade 1: Mild NPDR", "desc": "Subtle capillary microaneurysms near macular arcades."},
        {"file": "grade2_moderate_npdr.jpg", "grade": 2, "title": "Grade 2: Moderate NPDR", "desc": "From PPT expected output (Slide 11); hemorrhages and exudates."},
        {"file": "grade3_severe_npdr.jpg", "grade": 3, "title": "Grade 3: Severe NPDR", "desc": "Multiple quadrant blot hemorrhages and cotton-wool spots."},
        {"file": "grade4_proliferative_dr.jpg", "grade": 4, "title": "Grade 4: Proliferative DR", "desc": "Neovascularization of the disc (NVD) with vitreous hemorrhage risk."}
    ]

    for item in sample_files:
        path = os.path.join(SAMPLES_FOLDER, item["file"])
        if os.path.exists(path):
            img = cv2.imread(path)
            h, w = img.shape[:2]
            thumb = cv2.resize(img, (120, 120))
            samples.append({
                "id": item["file"],
                "filename": item["file"],
                "grade": item["grade"],
                "title": item["title"],
                "description": item["desc"],
                "dimensions": f"{w} × {h}",
                "thumbnail_b64": image_to_base64(thumb),
                "download_url": f"/api/samples/{item['file']}"
            })

    return jsonify({"samples": samples}), 200

@app.route("/api/samples/<filename>", methods=["GET"])
def serve_sample(filename):
    """Serves a sample image file."""
    return send_from_directory(SAMPLES_FOLDER, secure_filename(filename))

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """
    Module 1: Retinal Image Acquisition
    Uploads and validates retinal image format, dimensions, quality, and fundus characteristics.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file format. Please upload JPG, JPEG, or PNG retinal fundus images."
        }), 400

    # Save to uploads directory with unique ID
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_id = str(uuid.uuid4())[:8]
    clean_name = secure_filename(file.filename)
    stored_filename = f"{unique_id}_{clean_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
    file.save(save_path)

    # Read and inspect with OpenCV
    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        os.remove(save_path)
        return jsonify({"error": "Invalid image file. Unable to decode image data."}), 400

    h, w, c = img_bgr.shape
    file_size_bytes = os.path.getsize(save_path)

    # Inspect image quality
    quality_report = check_image_quality(img_bgr)

    # Convert preview to base64
    preview_b64 = image_to_base64(img_bgr)

    return jsonify({
        "success": True,
        "message": "Retinal image successfully acquired and verified.",
        "image_id": stored_filename,
        "original_filename": clean_name,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "dimensions": {"width": w, "height": h, "channels": c},
        "preview_b64": preview_b64,
        "quality_report": quality_report,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }), 200

@app.route("/api/select_sample", methods=["POST"])
def select_sample_image():
    """Copies a pre-existing sample into uploads for active analysis."""
    data = request.get_json() or {}
    sample_file = data.get("sample_file")
    if not sample_file:
        return jsonify({"error": "No sample_file provided."}), 400

    src_path = os.path.join(SAMPLES_FOLDER, secure_filename(sample_file))
    if not os.path.exists(src_path):
        return jsonify({"error": f"Sample image {sample_file} not found."}), 404

    img_bgr = cv2.imread(src_path)
    h, w, c = img_bgr.shape
    unique_id = str(uuid.uuid4())[:8]
    stored_filename = f"{unique_id}_{sample_file}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
    cv2.imwrite(save_path, img_bgr)

    file_size_bytes = os.path.getsize(save_path)
    quality_report = check_image_quality(img_bgr)
    preview_b64 = image_to_base64(img_bgr)

    return jsonify({
        "success": True,
        "message": f"Sample image '{sample_file}' selected for analysis.",
        "image_id": stored_filename,
        "original_filename": sample_file,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "dimensions": {"width": w, "height": h, "channels": c},
        "preview_b64": preview_b64,
        "quality_report": quality_report,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }), 200

@app.route("/api/preprocess", methods=["POST"])
def preprocess_image():
    """
    Module 2: Retinal Image Preprocessing
    Executes OpenCV pipeline: crop dark borders, resize 224x224, normalize, CLAHE, augmentations.
    """
    data = request.get_json() or {}
    image_id = data.get("image_id")

    if not image_id:
        return jsonify({"error": "Missing image_id parameter."}), 400

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_id))
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file '{image_id}' not found in server cache."}), 404

    try:
        pipeline_result = run_full_preprocessing_pipeline(image_path, output_dir=app.config["UPLOAD_FOLDER"])
        return jsonify({
            "success": True,
            "image_id": image_id,
            "pipeline": pipeline_result
        }), 200
    except Exception as e:
        return jsonify({"error": f"Preprocessing pipeline failed: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Module 3: CNN-Based Classification
    Takes an image, runs preprocessing if needed, and predicts 5 DR severity grades.
    """
    data = request.get_json() or {}
    image_id = data.get("image_id")

    if not image_id:
        return jsonify({"error": "Missing image_id parameter."}), 400

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_id))
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file '{image_id}' not found in server cache."}), 404

    try:
        # Load and ensure 224x224
        pipeline_result = run_full_preprocessing_pipeline(image_path, output_dir=app.config["UPLOAD_FOLDER"])
        processed_filename = pipeline_result["saved_processed_filename"]
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        processed_img_224 = cv2.imread(processed_path)
        if processed_img_224 is None:
            # Fallback to in-memory resize
            raw_img = cv2.imread(image_path)
            processed_img_224 = cv2.resize(raw_img, (224, 224))

        # Predict
        prediction_result = predict_retinal_image(processed_img_224, image_name=image_id)

        # Medical disclaimer text strictly mandated in project prompt Section 10
        disclaimer = (
            "This application is an academic Computer Vision project developed for educational and research "
            "demonstration purposes. Its output is not a medical diagnosis and must not replace examination "
            "or advice from a qualified ophthalmologist."
        )

        return jsonify({
            "success": True,
            "image_id": image_id,
            "prediction": prediction_result,
            "preprocessing_metadata": pipeline_result["metadata"],
            "processed_image_b64": pipeline_result["final_processed_b64"],
            "original_image_b64": pipeline_result["original_b64"],
            "disclaimer": disclaimer,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 200

    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

@app.route("/api/model/status", methods=["GET"])
def model_status():
    """Check whether trained model weights are loaded."""
    has_model = is_model_available()
    return jsonify({
        "trained_weights_available": has_model,
        "mode": "Trained CNN Inference" if has_model else "Demonstration Mode",
        "expected_input_shape": "224 × 224 × 3",
        "output_classes": 5,
        "classes": [DR_GRADES[i]["name"] for i in range(5)],
        "weights_path": "backend/model/dr_model.keras",
        "status_description": (
            "Model weights are active and performing direct tensor evaluations."
            if has_model else
            "Demonstration Mode active. No model weights found at backend/model/dr_model.keras. "
            "Train with 'python training/train.py' to generate model weights."
        )
    }), 200

@app.route("/api/model/metrics", methods=["GET"])
def model_metrics():
    """Return available evaluation metrics or placeholders."""
    metrics_data = get_model_evaluation_metrics()
    return jsonify(metrics_data), 200

@app.route("/api/download/<filename>", methods=["GET"])
def download_file(filename):
    """Download processed or uploaded image."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], secure_filename(filename), as_attachment=True)

if __name__ == "__main__":
    print("[Server] Starting Diabetic Retinopathy Detection Flask Backend on port 5000...")
    app.run(host="127.0.0.1", port=5000, debug=True)
