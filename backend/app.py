import os
import threading
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fiftyone as fo
from fiftyone import Sample, Classification
from PIL import Image
from dotenv import load_dotenv

# ----------------------------
# Load MongoDB credentials
# ----------------------------
load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

# ----------------------------
# CONFIG
# ----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATASET_NAME = "MyDataset"

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# FiftyOne dataset (persistent)
# ----------------------------
if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"📂 Loaded existing dataset: {DATASET_NAME}")
else:
    dataset = fo.Dataset(DATASET_NAME)
    print(f"✨ Created new dataset: {DATASET_NAME}")

# ----------------------------
# Assign random demo labels (only if missing)
# ----------------------------
def assign_demo_labels(ds):
    weld_shapes = ["round", "square", "irregular"]
    noise_types = ["low_noise", "medium_noise", "high_noise"]
    colors = ["red", "blue", "green"]

    for sample in ds:
        if not sample.tags:  # Only assign if tags empty
            chosen_labels = [
                random.choice(weld_shapes),
                random.choice(noise_types),
                random.choice(colors),
            ]
            sample.tags.extend(chosen_labels)
            sample.save()

assign_demo_labels(dataset)
print("✅ Demo labels ensured!")

# ----------------------------
# Add images from folders automatically (only missing ones)
# ----------------------------
def add_folder_images(base_path, split):
    for label in ["good", "bad"]:
        folder_path = os.path.join(base_path, split, "images", label)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(folder_path, filename)
                if not any(s.filepath == filepath for s in dataset):
                    sample = Sample(
                        filepath=filepath, 
                        ground_truth=Classification(label=label)
                    )
                    dataset.add_sample(sample)
    assign_demo_labels(dataset)

add_folder_images("dataset", "train")
add_folder_images("dataset", "val")

# ----------------------------
# Launch FiftyOne
# ----------------------------
def start_fiftyone():
    fo.launch_app(dataset, port=5151, remote=True, address="127.0.0.1")

threading.Thread(target=start_fiftyone, daemon=True).start()
print("✅ FiftyOne launching on http://127.0.0.1:5151")

# ----------------------------
# Serve images for React
# ----------------------------
@app.route("/dataset/<path:filename>")
def serve_dataset_image(filename):
    return send_from_directory("dataset", filename)

@app.route("/list_images")
def list_images():
    folder = request.args.get("folder")  # e.g., "train/images/good"
    folder_path = os.path.join("dataset", folder)
    if not os.path.exists(folder_path):
        return jsonify([])
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return jsonify(files)

# ----------------------------
# Upload route
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return {"error": "No file part"}, 400
    file = request.files["file"]
    if file.filename == "":
        return {"error": "No selected file"}, 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if not any(s.filepath == filepath for s in dataset):
        sample = Sample(filepath=filepath, ground_truth=Classification(label="unlabeled"))
        dataset.add_sample(sample)
        assign_demo_labels(dataset)

    return {"message": "File uploaded", "filename": file.filename, "label": "unlabeled"}

# ----------------------------
# Stats route
# ----------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    total_size = sum(os.path.getsize(s.filepath) for s in dataset) / 1e6
    return jsonify({
        "active_users": 1,
        "total_datasets": 10,
        "api_calls_today": 120,
        "storage_used": f"{total_size:.2f} MB"
    })

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
