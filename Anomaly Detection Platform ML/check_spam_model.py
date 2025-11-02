import joblib
import os

# Load the spam model
model_path = os.path.join("backend", "saved_models", "spam_model.joblib")
model = joblib.load(model_path)

print("Spam model loaded")
# Check if it has feature_names_in_
if hasattr(model, 'feature_names_in_'):
    print("Spam model feature names:", model.feature_names_in_)
else:
    print("Spam model does not have feature_names_in_")
