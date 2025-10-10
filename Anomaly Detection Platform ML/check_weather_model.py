import joblib
import os

# Load the weather model
model_path = os.path.join("backend", "saved_models", "weather_model.joblib")
model = joblib.load(model_path)

# Check if it has feature_names_in_
if hasattr(model, 'feature_names_in_'):
    print("Weather model feature names:", model.feature_names_in_)
else:
    print("Weather model does not have feature_names_in_")
