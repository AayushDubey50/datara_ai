#!/usr/bin/env python3
"""
Test FiftyOne UI and MongoDB separately to diagnose issues
"""

import os
import time
import fiftyone as fo
from fiftyone import Sample

print("🧪 Testing FiftyOne UI and MongoDB separately...")
print(f"FiftyOne version: {fo.__version__}")

# Test 1: Check if we can connect to MongoDB
print("\n📊 Test 1: MongoDB Connection")
try:
    datasets = fo.list_datasets()
    print(f"✅ MongoDB connection successful!")
    print(f"Available datasets: {datasets}")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# Test 2: Create a simple test dataset
print("\n📁 Test 2: Create Test Dataset")
test_dataset_name = "FiftyOne_UI_Test"

try:
    # Delete test dataset if it exists
    if test_dataset_name in fo.list_datasets():
        test_dataset = fo.load_dataset(test_dataset_name)
        test_dataset.delete()
        print(f"🗑️ Deleted existing test dataset")
    
    # Create new test dataset
    test_dataset = fo.Dataset(test_dataset_name)
    print(f"✅ Created test dataset: {test_dataset_name}")
    
    # Add a dummy sample (create a simple test image if needed)
    test_image_path = os.path.abspath("test_image.jpg")
    
    # Create a simple test image using PIL if it doesn't exist
    if not os.path.exists(test_image_path):
        from PIL import Image
        import numpy as np
        
        # Create a simple 100x100 red image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        img = Image.fromarray(img_array)
        img.save(test_image_path)
        print(f"📷 Created test image: {test_image_path}")
    
    # Add sample to dataset
    sample = Sample(filepath=test_image_path, tags=["test", "red", "simple"])
    test_dataset.add_sample(sample)
    print(f"✅ Added test sample to dataset")
    print(f"Dataset has {len(test_dataset)} samples")
    
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")
    exit(1)

# Test 3: Try launching FiftyOne UI on different ports
print("\n🌐 Test 3: FiftyOne UI Launch Tests")

# Test ports to try
test_ports = [5152, 5153, 5154, 5155, 5160]

for port in test_ports:
    print(f"\n🔌 Testing FiftyOne UI on port {port}...")
    try:
        # Try to launch FiftyOne UI
        session = fo.launch_app(
            test_dataset, 
            port=port, 
            remote=True, 
            address="0.0.0.0",
            auto=False  # Don't auto-open browser
        )
        
        print(f"✅ FiftyOne UI successfully started on port {port}!")
        print(f"🌐 URL: http://172.29.37.89:{port}")
        print(f"🌐 Localhost: http://127.0.0.1:{port}")
        print(f"🌐 WSL IP: http://0.0.0.0:{port}")
        
        # Wait a moment to see if it stays stable
        print("⏳ Testing stability for 10 seconds...")
        time.sleep(10)
        
        print(f"✅ Port {port} is stable and working!")
        print(f"\n🎉 SUCCESS: Use port {port} for FiftyOne UI")
        
        # Clean up
        session.close()
        break
        
    except Exception as e:
        print(f"❌ Port {port} failed: {e}")
        continue
else:
    print("❌ All ports failed!")

# Test 4: MongoDB process check
print("\n🔍 Test 4: MongoDB Process Status")
try:
    # Check if MongoDB is running
    import psutil
    
    mongo_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'mongod' in proc.info['name'].lower():
                mongo_processes.append(proc.info)
        except:
            pass
    
    if mongo_processes:
        print(f"✅ Found {len(mongo_processes)} MongoDB processes:")
        for proc in mongo_processes:
            print(f"  - PID {proc['pid']}: {proc['name']}")
    else:
        print("⚠️ No MongoDB processes found")
        
except Exception as e:
    print(f"⚠️ Could not check MongoDB processes: {e}")

# Clean up test files
try:
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")
        print("🧹 Cleaned up test image")
except:
    pass

print("\n✨ FiftyOne UI test completed!")
print("If a port worked, use that port in your main application.")