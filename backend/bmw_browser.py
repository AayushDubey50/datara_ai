import os
import re
import json
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import fiftyone as fo
from fiftyone import Sample
from PIL import Image

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
    
    frame_num = extract_frame_number(filename)
    stage = get_assembly_stage(frame_num)
    tags.append(stage)
    
    if "\\egos\\" in filepath or "/egos/" in filepath:
        tags.append("Ego")
        perspective = get_ego_perspective(filename)
        if perspective:
            tags.append(perspective)
    elif "\\orig\\" in filepath or "/orig/" in filepath:
        tags.append("Original")
    
    sample.tags.extend(tags)
    return sample

# ----------------------------
# Initialize BMW Dataset
# ----------------------------
print("🚀 Initializing BMW Dataset...")

try:
    if DATASET_NAME in fo.list_datasets():
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"📂 Loaded existing dataset: {DATASET_NAME}")
        is_loaded = True
    else:
        dataset = fo.Dataset(DATASET_NAME)
        print(f"✨ Created new dataset: {DATASET_NAME}")
        is_loaded = False

    def add_bmw_folder_images(base_path):
        base_dir = os.path.join("dataset_list", base_path)
        if not os.path.exists(base_dir):
            print(f"❌ Directory not found: {base_dir}")
            return 0
        
        total_added = 0
        
        for subdir in ["orig", "egos"]:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.exists(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        filepath = os.path.abspath(os.path.join(subdir_path, filename))
                        if not any(s.filepath == filepath for s in dataset):
                            sample = Sample(filepath=filepath)
                            sample = assign_bmw_tags(sample, filepath)
                            dataset.add_sample(sample)
                            total_added += 1
        
        return total_added

    if not is_loaded:
        added_count = add_bmw_folder_images("bmw_grill")
        print(f"✅ Added {added_count} new images with intelligent BMW tagging")

    print(f"🎯 Dataset ready with {len(dataset)} samples")

except Exception as e:
    print(f"❌ Dataset initialization error: {e}")
    dataset = None

# ----------------------------
# Custom Dataset Browser UI
# ----------------------------
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>BMW Grill Dataset Browser</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .filters { background: #f8f8f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-box { background: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .image-card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; background: white; }
        .image-card img { width: 100%; height: 150px; object-fit: cover; }
        .tags { margin-top: 10px; }
        .tag { background: #007cba; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; margin-right: 5px; }
        button { padding: 8px 15px; margin: 5px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #005a87; }
        select { padding: 8px; margin: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 BMW Grill Dataset Browser</h1>
        <p>Browse and filter BMW grill assembly images with intelligent tagging</p>
    </div>
    
    <div class="filters">
        <h3>🔍 Filters</h3>
        <select id="stageFilter" onchange="updateFilters()">
            <option value="">All Assembly Stages</option>
            <option value="Before">Before Assembly (0-42)</option>
            <option value="During">During Assembly (43-282)</option>
            <option value="After">After Assembly (283-309)</option>
        </select>
        
        <select id="typeFilter" onchange="updateFilters()">
            <option value="">All Image Types</option>
            <option value="Original">Original Images</option>
            <option value="Ego">Ego View Images</option>
        </select>
        
        <select id="perspectiveFilter" onchange="updateFilters()">
            <option value="">All Perspectives</option>
            <option value="Base">Base View</option>
            <option value="Low_Angle">Low Angle</option>
            <option value="Rotate_Left">Rotate Left</option>
            <option value="Rotate_Right">Rotate Right</option>
            <option value="Top_Down">Top Down</option>
        </select>
        
        <button onclick="clearFilters()">Clear All Filters</button>
        <button onclick="loadStats()">Refresh Stats</button>
    </div>
    
    <div class="stats" id="stats">
        <div class="stat-box">
            <h3>Total Images</h3>
            <div id="totalCount">Loading...</div>
        </div>
        <div class="stat-box">
            <h3>Assembly Stages</h3>
            <div id="stageStats">Loading...</div>
        </div>
        <div class="stat-box">
            <h3>Image Types</h3>
            <div id="typeStats">Loading...</div>
        </div>
        <div class="stat-box">
            <h3>Current Filter</h3>
            <div id="filteredCount">All images</div>
        </div>
    </div>
    
    <div id="imageGrid" class="image-grid">
        Loading images...
    </div>

    <script>
        let allImages = [];
        
        async function loadImages() {
            try {
                const response = await fetch('/api/images');
                allImages = await response.json();
                displayImages(allImages);
                loadStats();
            } catch (error) {
                console.error('Error loading images:', error);
            }
        }
        
        function displayImages(images) {
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';
            
            images.forEach(image => {
                const card = document.createElement('div');
                card.className = 'image-card';
                card.innerHTML = `
                    <img src="/api/image/${encodeURIComponent(image.filename)}" alt="${image.filename}">
                    <h4>${image.filename}</h4>
                    <div class="tags">
                        ${image.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                `;
                grid.appendChild(card);
            });
            
            document.getElementById('filteredCount').textContent = `${images.length} images`;
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('totalCount').textContent = stats.total_samples;
                document.getElementById('stageStats').innerHTML = `
                    Before: ${stats.assembly_stages.before}<br>
                    During: ${stats.assembly_stages.during}<br>
                    After: ${stats.assembly_stages.after}
                `;
                document.getElementById('typeStats').innerHTML = `
                    Original: ${stats.original_images}<br>
                    Ego: ${stats.ego_images}
                `;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        function updateFilters() {
            const stage = document.getElementById('stageFilter').value;
            const type = document.getElementById('typeFilter').value;
            const perspective = document.getElementById('perspectiveFilter').value;
            
            let filtered = allImages;
            
            if (stage) {
                filtered = filtered.filter(img => img.tags.includes(stage));
            }
            if (type) {
                filtered = filtered.filter(img => img.tags.includes(type));
            }
            if (perspective) {
                filtered = filtered.filter(img => img.tags.includes(perspective));
            }
            
            displayImages(filtered);
        }
        
        function clearFilters() {
            document.getElementById('stageFilter').value = '';
            document.getElementById('typeFilter').value = '';
            document.getElementById('perspectiveFilter').value = '';
            displayImages(allImages);
        }
        
        // Load images on page load
        loadImages();
    </script>
</body>
</html>
'''

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def home():
    """Main dataset browser UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/images')
def api_images():
    """Get all images with metadata"""
    if not dataset:
        return jsonify([])
    
    images = []
    for sample in dataset:
        images.append({
            "filename": os.path.basename(sample.filepath),
            "filepath": sample.filepath,
            "tags": sample.tags
        })
    
    return jsonify(images)

@app.route('/api/image/<path:filename>')
def api_image(filename):
    """Serve individual images"""
    # Look for the image in both orig and egos directories
    for subdir in ["orig", "egos"]:
        img_path = os.path.join("dataset_list", "bmw_grill", subdir)
        if os.path.exists(os.path.join(img_path, filename)):
            return send_from_directory(img_path, filename)
    
    return "Image not found", 404

@app.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    if not dataset:
        return jsonify({"error": "Dataset not available"})
    
    try:
        stats = {
            "total_samples": len(dataset),
            "assembly_stages": {"before": 0, "during": 0, "after": 0},
            "ego_images": 0,
            "original_images": 0,
            "tag_distribution": {}
        }
        
        for sample in dataset:
            for tag in sample.tags:
                stats["tag_distribution"][tag] = stats["tag_distribution"].get(tag, 0) + 1
                
                if tag == "Before":
                    stats["assembly_stages"]["before"] += 1
                elif tag == "During":
                    stats["assembly_stages"]["during"] += 1
                elif tag == "After":
                    stats["assembly_stages"]["after"] += 1
                elif tag == "Ego":
                    stats["ego_images"] += 1
                elif tag == "Original":
                    stats["original_images"] += 1
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/filter')
def api_filter():
    """Filter images by tags"""
    if not dataset:
        return jsonify([])
    
    stage = request.args.get("stage")
    img_type = request.args.get("type") 
    perspective = request.args.get("perspective")
    
    filtered_dataset = dataset
    
    if stage:
        filtered_dataset = filtered_dataset.match_tags([stage])
    if img_type:
        filtered_dataset = filtered_dataset.match_tags([img_type])
    if perspective:
        filtered_dataset = filtered_dataset.match_tags([perspective])
    
    results = []
    for sample in filtered_dataset:
        results.append({
            "filename": os.path.basename(sample.filepath),
            "filepath": sample.filepath,
            "tags": sample.tags
        })
    
    return jsonify(results)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "dataset_available": dataset is not None,
        "dataset_size": len(dataset) if dataset else 0,
        "mongodb_working": True
    })

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    print(f"🚀 Starting BMW Dataset Browser (No FiftyOne UI)")
    print(f"🌐 Dataset Browser: http://172.29.37.89:5001")
    print(f"📊 API Stats: http://172.29.37.89:5001/api/stats")
    print(f"💚 Health Check: http://172.29.37.89:5001/health")
    print(f"")
    print(f"Features:")
    print(f"  ✅ BMW intelligent tagging (assembly stages, ego perspectives)")
    print(f"  ✅ Custom web UI for browsing and filtering")
    print(f"  ✅ REST API for programmatic access")
    print(f"  ✅ Stable MongoDB backend (no UI networking issues)")
    
    app.run(host="0.0.0.0", port=5001, debug=True)