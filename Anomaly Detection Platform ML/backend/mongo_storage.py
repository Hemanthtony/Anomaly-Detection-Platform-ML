from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs
import os

# Use local MongoDB URI for cost-free local development
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "anomaly_detection"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# GridFS for storing large files like trained models
fs = gridfs.GridFS(db)

def save_model_file(file_data, filename):
    # Remove existing file with same name
    existing = db.fs.files.find_one({"filename": filename})
    if existing:
        fs.delete(existing["_id"])
    file_id = fs.put(file_data, filename=filename)
    return str(file_id)

def get_model_file(filename):
    file = fs.find_one({"filename": filename})
    if file:
        return file.read()
    return None

def save_user_data(user_data):
    # user_data is a dict
    users = db.users
    result = users.insert_one(user_data)
    return str(result.inserted_id)

def get_user_data(user_id):
    users = db.users
    user = users.find_one({"_id": ObjectId(user_id)})
    if user:
        user["_id"] = str(user["_id"])
    return user
