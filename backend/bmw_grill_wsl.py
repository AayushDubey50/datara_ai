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
# Load MongoDB credentials - Use embedded MongoDB instead of Atlas for WSL
# ----------------------------
load_dotenv()

# Remove external MongoDB configuration to use embedded MongoDB
# This avoids the DNS/networking issues in WSL
if 'FIFTYONE_DATABASE_URI' in os.environ:
    del os.environ['FIFTYONE_DATABASE_URI']
if 'FIFTYONE_DATABASE_NAME' in os.environ:
    del os.environ['FIFTYONE_DATABASE_NAME']

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
        
    perspective_part = parts[1].split(".")[0]  # Remove file extension
    
    # Map common perspective names to clean labels
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
    
    # Tag 1: Assembly Stage
    stage = get_assembly_stage(frame_num)
    tags.append(stage)
    
    # Tag 2: Image Type (based on directory)
    if "\\egos\\" in filepath or "/egos/" in filepath:
        tags.append("Ego")
        
        # Tag 3: Perspective for ego images
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
try:
    if DATASET_NAME in fo.list_datasets():
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"📂 Loaded existing dataset: {DATASET_NAME}")
        is_loaded = True
    else:
        dataset = fo.Dataset(DATASET_NAME)
        print(f"✨ Created new dataset: {DATASET_NAME}")
        is_loaded = False
except Exception as e:
    print(f"❌ Dataset error: {e}")
    dataset = fo.Dataset(DATASET_NAME)
    is_loaded = False

# ----------------------------
# Add BMW images with intelligent tagging
# ----------------------------
def add_bmw_folder_images(base_path):
    """Add BMW images from folder structure with intelligent tagging"""
    base_dir = os.path.join("dataset_list", base_path)
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return 0
    
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
    
    return total_added

# Load BMW images if dataset is new
if not is_loaded:
    added_count = add_bmw_folder_images("bmw_grill")
    print(f"✅ Added {added_count} new images with intelligent BMW tagging")

print(f"🎯 Dataset now has {len(dataset)} samples total")

# ----------------------------
# Launch FiftyOne with WSL-compatible settings
# ----------------------------
def start_fiftyone():
    try:
        # Use port 5152 to avoid conflicts and 0.0.0.0 for WSL networking
        fo.launch_app(dataset, port=5152, remote=True, address="0.0.0.0")
        print("✅ FiftyOne launched successfully!")
    except Exception as e:
        print(f"⚠️ FiftyOne launch warning: {e}")

threading.Thread(target=start_fiftyone, daemon=True).start()
print("🚀 FiftyOne launching on http://172.29.37.89:5152")

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
    for sample in dataset:
        for tag in sample.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    return jsonify({
        "total_samples": len(dataset),
        "tag_distribution": tag_counts,
        "assembly_stages": {
            "before": tag_counts.get("Before", 0),
            "during": tag_counts.get("During", 0), 
            "after": tag_counts.get("After", 0)
        },
        "image_types": {
            "ego": tag_counts.get("Ego", 0),
            "original": tag_counts.get("Original", 0)
        },
        "ego_perspectives": {
            "base": tag_counts.get("Base", 0),
            "low_angle": tag_counts.get("Low_Angle", 0),
            "rotate_left": tag_counts.get("Rotate_Left", 0),
            "rotate_right": tag_counts.get("Rotate_Right", 0),
            "top_down": tag_counts.get("Top_Down", 0)
        }
    })

@app.route("/bmw/filter")
def filter_images():
    """Filter BMW images by tags"""
    stage = request.args.get("stage")  # Before, During, After
    img_type = request.args.get("type")  # Ego, Original
    perspective = request.args.get("perspective")  # Base, Low_Angle, etc.
    
    filtered_samples = dataset
    
    if stage:
        filtered_samples = filtered_samples.match_tags([stage])
    if img_type:
        filtered_samples = filtered_samples.match_tags([img_type])
    if perspective:
        filtered_samples = filtered_samples.match_tags([perspective])
    
    results = []
    for sample in filtered_samples:
        results.append({
            "filepath": sample.filepath,
            "filename": os.path.basename(sample.filepath),
            "tags": sample.tags
        })
    
    return jsonify({
        "total_filtered": len(results),
        "images": results
    })

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

    # Add to FiftyOne dataset with BMW tagging
    if not any(s.filepath == filepath for s in dataset):
        sample = Sample(filepath=filepath)
        sample = assign_bmw_tags(sample, filepath)
        dataset.add_sample(sample)

    return {"message": "File uploaded with BMW tags", "filename": file.filename}

# ----------------------------
# Stats route
# ----------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        ego_count = len([s for s in dataset if "Ego" in s.tags])
        orig_count = len([s for s in dataset if "Original" in s.tags])
        before_count = len([s for s in dataset if "Before" in s.tags])
        during_count = len([s for s in dataset if "During" in s.tags])
        after_count = len([s for s in dataset if "After" in s.tags])
        
        return jsonify({
            "total_samples": len(dataset),
            "ego_images": ego_count,
            "original_images": orig_count,
            "assembly_stages": {
                "before": before_count,
                "during": during_count,
                "after": after_count
            },
            "dataset_name": DATASET_NAME,
            "intelligent_tagging": "enabled",
            "fiftyone_ui": "http://172.29.37.89:5152",
            "tag_analysis": "http://172.29.37.89:5001/bmw/tags/analysis"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return jsonify({
        "message": "BMW Grill Dataset Server with Intelligent Tagging",
        "dataset": DATASET_NAME,
        "total_samples": len(dataset),
        "features": [
            "Assembly stage tagging (Before/During/After)",
            "Image type classification (Ego/Original)", 
            "Ego perspective identification",
            "FiftyOne UI integration",
            "Tag-based filtering"
        ],
        "endpoints": [
            "GET /bmw/tags/analysis - Tag distribution analysis",
            "GET /bmw/filter - Filter images by tags",
            "GET /stats - Dataset statistics",
            "POST /upload - Upload new images with auto-tagging"
        ],
        "ui_links": {
            "fiftyone": "http://172.29.37.89:5152",
            "api": "http://172.29.37.89:5001",
            "tag_analysis": "http://172.29.37.89:5001/bmw/tags/analysis"
        }
    })

# ----------------------------
# Run Flask with WSL-compatible settings
# ----------------------------
if __name__ == "__main__":
    print(f"🚀 Starting BMW Grill Dataset Server with Intelligent Tagging")
    print(f"📊 FiftyOne UI: http://172.29.37.89:5152") 
    print(f"🔧 Flask API: http://172.29.37.89:5001")
    print(f"🏷️  Tag Analysis: http://172.29.37.89:5001/bmw/tags/analysis")
    print(f"📈 Dataset Stats: http://172.29.37.89:5001/stats")
    
    # Use port 5001 (which works) and 0.0.0.0 for WSL networking
    app.run(host="0.0.0.0", port=5001, debug=True)