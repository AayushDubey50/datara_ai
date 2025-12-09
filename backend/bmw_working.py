#!/usr/bin/env python3
"""
Working BMW grill script with embedded MongoDB cleanup
"""
import os
import re
import threading
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Clean up MongoDB lock files before importing FiftyOne
def cleanup_mongodb():
    """Clean up MongoDB lock files that prevent startup"""
    lock_files = [
        os.path.expanduser("~/.fiftyone/var/lib/mongo/mongod.lock"),
        os.path.expanduser("~/.fiftyone/var/lib/mongo/WiredTiger.lock")
    ]
    
    for lock_file in lock_files:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                print(f"🧹 Removed lock file: {lock_file}")
        except Exception as e:
            print(f"⚠️ Could not remove {lock_file}: {e}")

# Clean MongoDB before importing FiftyOne
cleanup_mongodb()

import fiftyone as fo
from fiftyone import Sample, Classification

print("✅ FiftyOne imported successfully after cleanup")

# Load environment
load_dotenv()

# Config
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATASET_NAME = "BMW_GRILL"

# Flask app
app = Flask(__name__)
CORS(app)

# BMW Tagging Functions (same as before)
def extract_frame_number(filename):
    """Extract frame number from BMW image filename"""
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
        
    parts = filename.split("_ego_")
    if len(parts) < 2:
        return None
        
    perspective_part = parts[1].split(".")[0]
    
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
    
    # Tag 2: Image Type
    if "\\egos\\" in filepath or "/egos/" in filepath:
        tags.append("Ego")
        
        # Tag 3: Perspective
        perspective = get_ego_perspective(filename)
        if perspective:
            tags.append(perspective)
            
    elif "\\orig\\" in filepath or "/orig/" in filepath:
        tags.append("Original")
    
    # Apply tags
    sample.tags.extend(tags)
    print(f"📌 Tagged {filename}: {tags}")
    return sample

# FiftyOne dataset
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

def add_bmw_folder_images(base_path):
    """Add BMW images with tagging"""
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
                
                if not any(s.filepath == filepath for s in dataset):
                    sample = Sample(filepath=filepath)
                    sample = assign_bmw_tags(sample, filepath)
                    dataset.add_sample(sample)
                    total_added += 1
    
    return total_added

# Load BMW images
if not is_loaded:
    added_count = add_bmw_folder_images("bmw_grill")
    print(f"✅ Added {added_count} new images")

print(f"🎯 Dataset now has {len(dataset)} samples total")

# Launch FiftyOne
def start_fiftyone():
    time.sleep(2)  # Give time for setup
    fo.launch_app(dataset, port=5151, remote=True, address="127.0.0.1")

threading.Thread(target=start_fiftyone, daemon=True).start()
print("✅ FiftyOne launching on http://127.0.0.1:5151")

# Flask routes
@app.route("/datasets/<path:filename>")
def serve_dataset_image(filename):
    return send_from_directory(os.path.join("dataset_list", "bmw_grill"), filename)

@app.route("/list_images")
def list_images():
    folder = request.args.get("folder")
    folder_path = os.path.join("dataset_list", "bmw_grill", folder)
    if not os.path.exists(folder_path):
        return jsonify([])
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return jsonify(files)

@app.route("/bmw/tags/analysis")
def analyze_tags():
    """Analyze tag distribution"""
    tag_counts = {}
    for sample in dataset:
        for tag in sample.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    return jsonify({
        "total_samples": len(dataset),
        "tag_distribution": tag_counts
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        ego_count = len([s for s in dataset if "Ego" in s.tags])
        orig_count = len([s for s in dataset if "Original" in s.tags])
        
        return jsonify({
            "total_samples": len(dataset),
            "ego_images": ego_count,
            "original_images": orig_count,
            "dataset_name": DATASET_NAME
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting BMW Grill Dataset Server")
    print("📊 FiftyOne UI: http://127.0.0.1:5151") 
    print("🔧 Flask API: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)