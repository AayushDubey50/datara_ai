"""
Example script demonstrating how to access MongoDB database and retrieve image metadata.
This script shows how to connect to MongoDB, query images, and access their metadata.
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Load MongoDB credentials from .env
load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

if not db_password:
    print("❌ Error: MONGODB_PASSWORD not found in .env file")
    print("Please create a .env file with: MONGODB_PASSWORD=your_password")
    exit(1)

# Connect to MongoDB cluster
connection_string = f"mongodb+srv://rithviggolf:{db_password}@roboticdata.pqtfhwu.mongodb.net/"
client = MongoClient(connection_string)

# Access the FiftyOne database
db = client.get_database("fiftyone")
samples_collection = db["samples"]

print("✅ Connected to MongoDB")
print("Database: fiftyone")
print("Collection: samples\n")

# Get total number of images
total_images = samples_collection.count_documents({})
print(f"📊 Total images in database: {total_images}\n")

# Example 1: Get all images
print("=" * 60)
print("Example 1: List all images")
print("=" * 60)
all_samples = list(samples_collection.find({}).limit(10))  # Limit to first 10
for i, sample in enumerate(all_samples, 1):
    filepath = sample.get("filepath", "N/A")
    tags = sample.get("tags", [])
    ground_truth = sample.get("ground_truth", {})
    print(f"{i}. {Path(filepath).name}")
    print(f"   Path: {filepath}")
    print(f"   Tags: {tags}")
    if ground_truth:
        print(f"   Ground Truth: {ground_truth}")
    print()

# Example 2: Find images by tag
print("=" * 60)
print("Example 2: Find images by tag")
print("=" * 60)
images_with_tag = list(samples_collection.find({"tags": {"$in": ["round"]}}).limit(5))
print(f"Found {len(images_with_tag)} images with 'round' tag")
for sample in images_with_tag:
    print(f"  - {Path(sample.get('filepath', '')).name}")

# Example 3: Find images by ground truth label
print("\n" + "=" * 60)
print("Example 3: Find images by ground truth label")
print("=" * 60)
good_images = list(samples_collection.find({"ground_truth.label": "good"}).limit(5))
bad_images = list(samples_collection.find({"ground_truth.label": "bad"}).limit(5))
print(f"Good images: {len(good_images)}")
print(f"Bad images: {len(bad_images)}")

# Example 4: Get image statistics
print("\n" + "=" * 60)
print("Example 4: Database statistics")
print("=" * 60)
all_samples_stats = list(samples_collection.find({}))
tag_counts = {}
label_counts = {}

for sample in all_samples_stats:
    # Count tags
    for tag in sample.get("tags", []):
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Count labels
    label = sample.get("ground_truth", {}).get("label", "unlabeled")
    label_counts[label] = label_counts.get(label, 0) + 1

print("Tag distribution:")
for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {tag}: {count}")

print("\nLabel distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label}: {count}")

# Example 5: Get image file paths
print("\n" + "=" * 60)
print("Example 5: Get all image file paths")
print("=" * 60)
all_filepaths = [sample.get("filepath") for sample in samples_collection.find({}, {"filepath": 1})]
#                  if sample.get("filepath")]
# print(f"Total file paths: {len(all_filepaths)}")
# print("Sample paths:")
# for path in all_filepaths[:5]:
#     print(f"  - {path}")

# # Example 6: Check if image file exists
# print("\n" + "=" * 60)
# print("Example 6: Verify image files exist on filesystem")
# print("=" * 60)
# existing_count = 0
# missing_count = 0

# for sample in list(samples_collection.find({}).limit(10)):
#     filepath = sample.get("filepath")
#     if filepath:
#         if os.path.exists(filepath):
#             existing_count += 1
#         else:
#             missing_count += 1
#             print(f"  ⚠️  Missing: {filepath}")

# print(f"\nVerified: {existing_count} exist, {missing_count} missing (checked first 10)")

# print("\n" + "=" * 60)
# print("✅ MongoDB access examples complete")
# print("=" * 60)

# Close connection
client.close()

