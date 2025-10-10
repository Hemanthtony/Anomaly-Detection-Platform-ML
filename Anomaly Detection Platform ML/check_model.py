import joblib
model = joblib.load('backend/saved_models/network_model.joblib')
print(hasattr(model, 'feature_names_in_'))
if hasattr(model, 'feature_names_in_'):
    print(model.feature_names_in_)
else:
    print("No feature_names_in_")
