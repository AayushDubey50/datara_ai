from fastapi import APIRouter
from datetime import datetime, timedelta
import random
import os

router = APIRouter()

# === Simulated Data ===
def get_api_calls_last_7_days():
    today = datetime.now()
    return [
        {"date": (today - timedelta(days=i)).strftime("%Y-%m-%d"), "calls": random.randint(50, 200)}
        for i in range(6, -1, -1)
    ]

def get_recent_searches():
    return ["weld defect", "pipe crack", "good weld", "misalignment", "corrosion"]

@router.get("/api/dashboard-stats")
def get_dashboard_stats():
    # Replace these random values with real database queries later
    api_calls_today = random.randint(500, 1200)
    active_users = random.randint(10, 50)

    # Count total images in dataset/train/images (example)
    image_folder = os.path.join("dataset", "train", "images")
    total_images = sum(len(files) for _, _, files in os.walk(image_folder)) if os.path.exists(image_folder) else 0

    api_calls_last_7_days = get_api_calls_last_7_days()
    recent_searches = get_recent_searches()

    return {
        "apiCalls": api_calls_today,
        "activeUsers": active_users,
        "totalImages": total_images,
        "apiCallsLast7Days": api_calls_last_7_days,
        "recentSearches": recent_searches
    }
