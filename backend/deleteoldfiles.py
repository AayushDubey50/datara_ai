import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load MongoDB credentials from .env
load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")

# Connect to your MongoDB cluster
client = MongoClient(f"mongodb+srv://rithviggolf:{db_password}@roboticdata.pqtfhwu.mongodb.net/")

# Select the FiftyOne database and samples collection
db = client.get_database("fiftyone")
db["samples"].delete_many({})  # Deletes all images/documents
print("✅ All images deleted from MongoDB")
