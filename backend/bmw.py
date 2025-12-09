import os
import threading
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

# Try importing FiftyOne with fallback
try:
    import fiftyone as fo
    from fiftyone import Sample, Classification
    FIFTYONE_AVAILABLE = True
    print("✅ FiftyOne imported successfully")
except ImportError as e:
    FIFTYONE_AVAILABLE = False
    print(f"⚠️ FiftyOne not available: {e}")
    print("   Server will run without FiftyOne visualization")

load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATASET_NAME = "BMW_GRILL"

app = Flask(__name__)
CORS(app)

# Initialize dataset only if FiftyOne is available
dataset = None
is_loaded = False

if FIFTYONE_AVAILABLE:
    try:
        if DATASET_NAME in fo.list_datasets():
            dataset = fo.load_dataset(DATASET_NAME)
            print(f"📂 Loaded existing dataset: {DATASET_NAME}")
            is_loaded = True
        else:
            dataset = fo.Dataset(DATASET_NAME)
            print(f"✨ Created new dataset: {DATASET_NAME}")
    except Exception as e:
        print(f"❌ FiftyOne dataset error: {e}")
        dataset = None
else:
    print("📊 Running without FiftyOne dataset management")




# def assign_demo_labels(ds):
#     weld_shapes = ["round", "square", "irregular"]
#     noise_types = ["low_noise", "medium_noise", "high_noise"]
#     colors = ["red", "blue", "green"]

#     for sample in ds:
#         if not sample.tags:  # Only assign if tags empty
#             chosen_labels = [
#                 random.choice(weld_shapes),
#                 random.choice(noise_types),
#                 random.choice(colors),
#             ]
#             sample.tags.extend(chosen_labels)
#             sample.save()

# # assign_demo_labels(dataset)
# print("✅ Demo labels ensured!")




def add_folder_images(base_path):
    if not FIFTYONE_AVAILABLE or dataset is None:
        print("⚠️ Skipping dataset loading - FiftyOne not available")
        return
        
    base_dir = os.path.join("dataset_list", base_path)
    if not os.path.exists(base_dir):
        print(f"📁 Directory not found: {base_dir}")
        return
        
    orig_dir = os.path.join(base_dir, "orig")
    egos_dir = os.path.join(base_dir, "egos")
    
    # Load original images
    if os.path.exists(orig_dir):
        orig_files = [f for f in os.listdir(orig_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for file_path in orig_files:
            try:
                sample = Sample(
                    filepath=os.path.join(orig_dir, file_path),
                    tags=["exocentric"]
                )
                dataset.add_sample(sample)
            except Exception as e:
                print(f"⚠️ Error adding {file_path}: {e}")
        print(f"📸 Added {len(orig_files)} original images")
    
    # Load ego images  
    if os.path.exists(egos_dir):
        ego_files = [f for f in os.listdir(egos_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for file_path in ego_files:
            try:
                sample = Sample(
                    filepath=os.path.join(egos_dir, file_path),
                    tags=["egocentric"]
                )
                dataset.add_sample(sample)
            except Exception as e:
                print(f"⚠️ Error adding {file_path}: {e}")
        print(f"🎯 Added {len(ego_files)} ego images")

if not is_loaded:
    add_folder_images("bmw_grill")

if FIFTYONE_AVAILABLE and dataset:
    print(f"✅ Dataset has {len([s for s in dataset])} samples")
else:
    print("📊 Dataset management disabled - using file system only")



# ----------------------------
# Launch FiftyOne
# ----------------------------
def start_fiftyone():
    if not FIFTYONE_AVAILABLE or dataset is None:
        print("⚠️ FiftyOne not available - skipping launch")
        return
        
    try:
        fo.launch_app(dataset, port=5152, remote=True, address="127.0.0.1")
        print("✅ FiftyOne launched on http://127.0.0.1:5152")
    except Exception as e:
        print(f"⚠️ FiftyOne launch error (port 5152): {e}")
        try:
            # Try alternative port if 5152 is busy
            fo.launch_app(dataset, port=5153, remote=True, address="127.0.0.1") 
            print("✅ FiftyOne launched on http://127.0.0.1:5153")
        except Exception as e2:
            print(f"❌ FiftyOne failed to start on both ports: {e2}")

if FIFTYONE_AVAILABLE:
    threading.Thread(target=start_fiftyone, daemon=True).start()
    print("🚀 Attempting to launch FiftyOne...")
else:
    print("📊 FiftyOne visualization disabled")

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

    # Add to FiftyOne dataset if available
    if FIFTYONE_AVAILABLE and dataset:
        try:
            if not any(s.filepath == filepath for s in dataset):
                sample = Sample(filepath=filepath, ground_truth=Classification(label="unlabeled"))
                dataset.add_sample(sample)
        except Exception as e:
            print(f"⚠️ Error adding to dataset: {e}")

    return {"message": "File uploaded", "filename": file.filename, "label": "unlabeled"}

# ----------------------------
# Stats route
# ----------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    # Get file-based stats
    orig_dir = os.path.join("dataset_list", "bmw_grill", "orig")
    egos_dir = os.path.join("dataset_list", "bmw_grill", "egos")
    
    orig_count = len([f for f in os.listdir(orig_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(orig_dir) else 0
    egos_count = len([f for f in os.listdir(egos_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(egos_dir) else 0
    
    # Calculate storage if FiftyOne dataset available
    total_size_mb = 0
    if FIFTYONE_AVAILABLE and dataset:
        try:
            total_size = sum(os.path.getsize(s.filepath) for s in dataset if os.path.exists(s.filepath)) / 1e6
            total_size_mb = round(total_size, 2)
        except Exception as e:
            print(f"⚠️ Error calculating storage: {e}")
    
    return jsonify({
        "orig_frames": orig_count,
        "ego_frames": egos_count,
        "total_frames": orig_count + egos_count,
        "storage_used": f"{total_size_mb} MB" if total_size_mb > 0 else "Unknown",
        "fiftyone_enabled": FIFTYONE_AVAILABLE,
        "dataset_loaded": dataset is not None
    })

@app.route("/")
def index():
    return jsonify({
        "message": "BMW Grill Frame Management Server",
        "dataset": DATASET_NAME,
        "fiftyone_enabled": FIFTYONE_AVAILABLE,
        "endpoints": [
            "/list_orig_frames",
            "/list_ego_frames", 
            "/list_images?folder=orig",
            "/list_images?folder=egos",
            "/datasets/{filename}",
            "/stats",
            "/upload"
        ]
    })

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    print("✅ BMW Frame Management Server starting...")
    print(f"🚀 Flask server: http://127.0.0.1:5051")
    print(f"📂 Dataset directory: dataset_list/bmw_grill/")
    if FIFTYONE_AVAILABLE:
        print(f"👀 FiftyOne will attempt to start on port 5152 or 5153")
    app.run(host="127.0.0.1", port=5051, debug=True)
