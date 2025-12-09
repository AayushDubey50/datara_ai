import os
import re
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fiftyone as fo
from fiftyone import Sample, Classification
from PIL import Image
from dotenv import load_dotenv

# ----------------------------
# Load MongoDB credentials and configure FiftyOne
# ----------------------------
load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

# Configure FiftyOne to use external MongoDB Atlas instead of embedded MongoDB
os.environ['FIFTYONE_DATABASE_URI'] = f"mongodb+srv://rithviggolf:{db_password}@roboticdata.pqtfhwu.mongodb.net/"
os.environ['FIFTYONE_DATABASE_NAME'] = "fiftyone_bmw"

# ----------------------------
# CONFIG
# ----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATASET_NAME = "BMW_GRILL"

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# BMW Tagging Functions
# ----------------------------
def extract_frame_number(filename):
    """Extract frame number from BMW image filename"""
    # Pattern: bmw_grill_36.jpg or bmw_grill_36_ego_base.jpg
    match = re.search(r'bmw_grill_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_assembly_stage(frame_number):
    """Determine assembly stage based on frame number"""
    if frame_number is None:
        return "Unknown"
    
    if 0 <= frame_number <= 42:
        return "Before"
    elif 43 <= frame_number <= 282:
        return "During" 
    elif 283 <= frame_number <= 309:
        return "After"
    else:
        return "Unknown"

def get_ego_perspective(filename):
    """Extract ego perspective from filename"""
    if "_ego_" not in filename:
        return None
        
    # Extract text after "ego_"
    parts = filename.split("_ego_")
    if len(parts) < 2:
        return None
        
    # Get perspective part and remove file extension
    perspective_part = parts[1].split(".")[0]
    
    # Convert to proper format
    perspective_map = {
        "base": "Base",
        "low_angle": "Low_Angle", 
        "rotate_left": "Rotate_Left",
        "rotate_right": "Rotate_Right",
        "top_down": "Top_Down"
    }
    
    return perspective_map.get(perspective_part, perspective_part.replace("_", "_").title())

def assign_bmw_tags(sample, filepath):
    """Assign BMW-specific tags based on filepath and filename"""
    filename = os.path.basename(filepath)
    tags = []
    
    # Extract frame number
    frame_num = extract_frame_number(filename)
    
    # Tag 1: Assembly Stage (applies to all)
    stage = get_assembly_stage(frame_num)
    tags.append(stage)
    
    # Tag 2: Image Type (based on directory)
    if "\\egos\\" in filepath or "/egos/" in filepath:
        tags.append("Ego")
        
        # Tag 3: Perspective (only for ego images)
        perspective = get_ego_perspective(filename)
        if perspective:
            tags.append(perspective)
            
    elif "\\orig\\" in filepath or "/orig/" in filepath:
        tags.append("Original")
    
    # Apply tags to sample
    sample.tags.extend(tags)
    print(f"📌 Tagged {filename}: {tags}")
    return sample

# ----------------------------
# FiftyOne dataset (persistent)
# ----------------------------
if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"📂 Loaded existing dataset: {DATASET_NAME}")
    is_loaded = True
else:
    dataset = fo.Dataset(DATASET_NAME)
    print(f"✨ Created new dataset: {DATASET_NAME}")
    is_loaded = False

# ----------------------------
# Add BMW images with intelligent tagging
# ----------------------------
def add_bmw_folder_images(base_path):
    """Add images from BMW dataset folders with proper tagging"""
    base_dir = os.path.join("dataset_list", base_path)
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return
    
    total_added = 0
    
    # Process orig directory
    orig_dir = os.path.join(base_dir, "orig")
    if os.path.exists(orig_dir):
        print(f"📁 Processing original images from: {orig_dir}")
        for filename in os.listdir(orig_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.abspath(os.path.join(orig_dir, filename))
                
                # Check if sample already exists
                if not any(s.filepath == filepath for s in dataset):
                    sample = Sample(filepath=filepath)
                    sample = assign_bmw_tags(sample, filepath)
                    dataset.add_sample(sample)
                    total_added += 1
        
    # Process egos directory  
    egos_dir = os.path.join(base_dir, "egos")
    if os.path.exists(egos_dir):
        print(f"📁 Processing ego images from: {egos_dir}")
        for filename in os.listdir(egos_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.abspath(os.path.join(egos_dir, filename))
                
                # Check if sample already exists
                if not any(s.filepath == filepath for s in dataset):
                    sample = Sample(filepath=filepath)
                    sample = assign_bmw_tags(sample, filepath)
                    dataset.add_sample(sample)
                    total_added += 1
    
    print(f"✅ Added {total_added} BMW images to dataset")
    return total_added

# Load BMW images (only if not already loaded)
if not is_loaded:
    added_count = add_bmw_folder_images("bmw_grill")
    print(f"✅ Added {added_count} new images")

print(f"🎯 Dataset now has {len(dataset)} samples total")



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

# ----------------------------
# BMW-specific API endpoints
# ----------------------------
@app.route("/bmw/tags/analysis")
def analyze_tags():
    """Analyze tag distribution in BMW dataset"""
    tag_counts = {}
    stage_counts = {}
    perspective_counts = {}
    
    for sample in dataset:
        for tag in sample.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Categorize tags
            if tag in ["Before", "During", "After"]:
                stage_counts[tag] = stage_counts.get(tag, 0) + 1
            elif tag in ["Base", "Low_Angle", "Rotate_Left", "Rotate_Right", "Top_Down"]:
                perspective_counts[tag] = perspective_counts.get(tag, 0) + 1
    
    return jsonify({
        "total_samples": len(dataset),
        "all_tags": tag_counts,
        "assembly_stages": stage_counts,
        "ego_perspectives": perspective_counts
    })

@app.route("/bmw/tags/assign", methods=["POST"])
def assign_custom_tags():
    """Manually assign additional tags to specific samples"""
    sample_id = request.json.get("sample_id")
    tags = request.json.get("tags", [])
    
    try:
        sample = dataset[sample_id]
        sample.tags.extend(tags)
        sample.save()
        return jsonify({"success": True, "message": f"Added tags {tags} to sample"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ----------------------------
# Upload route (same as app.py)
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
        sample = assign_bmw_tags(sample, filepath)
        dataset.add_sample(sample)

    return {"message": "File uploaded", "filename": file.filename, "tags": sample.tags}

# ----------------------------
# Stats route
# ----------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        total_size = sum(os.path.getsize(s.filepath) for s in dataset if os.path.exists(s.filepath)) / 1e6
        
        # Count by image type
        ego_count = len([s for s in dataset if "Ego" in s.tags])
        orig_count = len([s for s in dataset if "Original" in s.tags])
        
        return jsonify({
            "total_samples": len(dataset),
            "ego_images": ego_count,
            "original_images": orig_count,
            "storage_used": f"{total_size:.2f} MB",
            "dataset_name": DATASET_NAME
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    print(f"🚀 Starting BMW Grill Dataset Server")
    print(f"📊 FiftyOne UI: http://127.0.0.1:5151") 
    print(f"🔧 Flask API: http://127.0.0.1:5000")
    print(f"🏷️  Tag Analysis: http://127.0.0.1:5000/bmw/tags/analysis")
    app.run(host="127.0.0.1", port=5000, debug=True)
