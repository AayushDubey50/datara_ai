import os
import threading
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

print("🚀 Starting BMW Test Server...")

# ----------------------------
# Serve images for React
# ----------------------------
@app.route("/datasets/<path:filename>")
def serve_dataset_image(filename):
    return send_from_directory(os.path.join("dataset_list", "bmw_grill"), filename)

@app.route("/list_images")
def list_images():
    folder = request.args.get("folder")  # e.g., "orig" or "egos"
    folder_path = os.path.join("dataset_list", "bmw_grill", folder)
    if not os.path.exists(folder_path):
        return jsonify([])
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return jsonify(files)

@app.route("/list_orig_frames")
def list_orig_frames():
    """List all original frames available"""
    orig_dir = os.path.join("dataset_list", "bmw_grill", "orig")
    if not os.path.exists(orig_dir):
        return jsonify([])
    
    frames = []
    for filename in os.listdir(orig_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            frames.append({
                "filename": filename,
                "path": f"orig/{filename}"
            })
    
    return jsonify(sorted(frames, key=lambda x: x["filename"]))

@app.route("/list_ego_frames")
def list_ego_frames():
    """List all ego frames available"""
    ego_dir = os.path.join("dataset_list", "bmw_grill", "egos")
    if not os.path.exists(ego_dir):
        return jsonify([])
    
    frames = []
    for filename in os.listdir(ego_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            frames.append({
                "filename": filename,
                "path": f"egos/{filename}"
            })
    
    return jsonify(sorted(frames, key=lambda x: x["filename"]))

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

    return {"message": "File uploaded", "filename": file.filename}

# ----------------------------
# Stats route
# ----------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    orig_dir = os.path.join("dataset_list", "bmw_grill", "orig")
    egos_dir = os.path.join("dataset_list", "bmw_grill", "egos")
    
    orig_count = len([f for f in os.listdir(orig_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(orig_dir) else 0
    egos_count = len([f for f in os.listdir(egos_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(egos_dir) else 0
    
    return jsonify({
        "orig_frames": orig_count,
        "ego_frames": egos_count,
        "total_frames": orig_count + egos_count,
        "server_status": "running"
    })

@app.route("/")
def index():
    return jsonify({
        "message": "BMW Frame Management Server",
        "endpoints": [
            "/list_orig_frames",
            "/list_ego_frames", 
            "/list_images?folder=orig",
            "/list_images?folder=egos",
            "/datasets/{filename}",
            "/stats"
        ]
    })

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    print("✅ Flask server starting on http://127.0.0.1:5051")
    print("📂 Dataset directory: dataset_list/bmw_grill/")
    app.run(host="127.0.0.1", port=5051, debug=True)