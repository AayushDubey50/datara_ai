#!/usr/bin/env python3
"""
Script to load BMW grill images into the existing FiftyOne dataset running on port 5151
"""

import os
import fiftyone as fo
from fiftyone import Sample

# Connect to the existing FiftyOne session
try:
    # List all available datasets
    datasets = fo.list_datasets()
    print("Available datasets:")
    for ds_name in datasets:
        print(f"  - {ds_name}")
    
    # Try to load the most recently used dataset or create BMW_GRILL
    if "BMW" in datasets:
        dataset = fo.load_dataset("BMW")
        print(f"📂 Loaded existing dataset: BMW")
    elif "MyDataset" in datasets:
        dataset = fo.load_dataset("MyDataset")
        print(f"📂 Loaded existing dataset: MyDataset")
        # Clear existing samples to replace with BMW images
        dataset.clear()
        print("🧹 Cleared existing samples")
    else:
        dataset = fo.Dataset("BMW_GRILL")
        print(f"✨ Created new dataset: BMW_GRILL")
    
    # Add BMW grill images
    base_dir = os.path.join("dataset_list", "bmw_grill")
    
    # Add original images
    orig_dir = os.path.join(base_dir, "orig")
    if os.path.exists(orig_dir):
        orig_count = 0
        for filename in os.listdir(orig_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.abspath(os.path.join(orig_dir, filename))
                # Check if sample already exists
                if not any(s.filepath == filepath for s in dataset):
                    sample = Sample(
                        filepath=filepath,
                        tags=["exocentric", "original", "bmw_grill"]
                    )
                    dataset.add_sample(sample)
                    orig_count += 1
        print(f"📸 Added {orig_count} original images")
    
    # Add ego images
    egos_dir = os.path.join(base_dir, "egos")
    if os.path.exists(egos_dir):
        ego_count = 0
        for filename in os.listdir(egos_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.abspath(os.path.join(egos_dir, filename))
                # Check if sample already exists
                if not any(s.filepath == filepath for s in dataset):
                    # Parse ego view type from filename
                    ego_type = "base"
                    if "_low_angle" in filename:
                        ego_type = "low_angle"
                    elif "_rotate_left" in filename:
                        ego_type = "rotate_left"
                    elif "_rotate_right" in filename:
                        ego_type = "rotate_right"
                    elif "_top_down" in filename:
                        ego_type = "top_down"
                    
                    sample = Sample(
                        filepath=filepath,
                        tags=["egocentric", ego_type, "bmw_grill"]
                    )
                    dataset.add_sample(sample)
                    ego_count += 1
        print(f"🎯 Added {ego_count} ego images")
    
    # Save the dataset
    dataset.persistent = True
    print(f"✅ Dataset has {len(dataset)} total samples")
    print(f"🌐 View dataset at: http://localhost:5151")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure FiftyOne is running on port 5151")