import os
import pandas as pd
import joblib
from models.network_anomaly import train_network_model
from models.weather_anomaly import train_weather_model
from models.spam_detection import train_spam_model
import kagglehub

MODEL_DIR = "saved_models"
DATASET_DIR = "datasets"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def download_network_dataset():
    """Download network anomaly detection dataset from Kaggle"""
    try:
        path = kagglehub.model_download("malkasasbeh/network-anomaly-detection-dataset")
        print("Network dataset downloaded to:", path)
        return path
    except Exception as e:
        print(f"Error downloading network dataset: {e}")
        return None

def download_weather_dataset():
    """Download weather anomaly detection dataset"""
    # Placeholder - add actual dataset download code
    print("Weather dataset download not implemented yet.")
    return None

def download_spam_dataset():
    """Download spam detection dataset"""
    # Placeholder - add actual dataset download code
    print("Spam dataset download not implemented yet.")
    return None

def train_network_model_large():
    """Train network anomaly detection model with large dataset"""
    print("Training network anomaly detection model...")

    # Download dataset
    dataset_path = download_network_dataset()
    if not dataset_path:
        print("Failed to download network dataset. Using sample data.")
        dataset_path = "../sample_network.csv"

    # Load data
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded network dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading network dataset: {e}")
        return False

    # Train model
    try:
        model = train_network_model(df)
        model_path = os.path.join(MODEL_DIR, "network_model.joblib")
        joblib.dump(model, model_path)
        print(f"Network model trained and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error training network model: {e}")
        return False

def train_weather_model_large():
    """Train weather anomaly detection model with large dataset"""
    print("Training weather anomaly detection model...")

    # Download dataset
    dataset_path = download_weather_dataset()
    if not dataset_path:
        print("Failed to download weather dataset. Using sample data.")
        dataset_path = "../sample_weather.csv"

    # Load data
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded weather dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading weather dataset: {e}")
        return False

    # Train model
    try:
        model = train_weather_model(df)
        model_path = os.path.join(MODEL_DIR, "weather_model.joblib")
        joblib.dump(model, model_path)
        print(f"Weather model trained and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error training weather model: {e}")
        return False

def train_spam_model_large():
    """Train spam detection model with large dataset"""
    print("Training spam detection model...")

    # Download dataset
    dataset_path = download_spam_dataset()
    if not dataset_path:
        print("Failed to download spam dataset. Using sample data.")
        dataset_path = "../sample_spam.csv"

    # Load data
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded spam dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading spam dataset: {e}")
        return False

    # Train model
    try:
        model = train_spam_model(df)
        model_path = os.path.join(MODEL_DIR, "spam_model.joblib")
        joblib.dump(model, model_path)
        print(f"Spam model trained and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error training spam model: {e}")
        return False

def train_all_models():
    """Train all three models with large datasets"""
    print("Starting training of all anomaly detection models...")

    results = {
        "network": train_network_model_large(),
        "weather": train_weather_model_large(),
        "spam": train_spam_model_large()
    }

    print("\nTraining Results:")
    for model_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{model_type.capitalize()} model: {status}")

    return results

if __name__ == "__main__":
    train_all_models()
