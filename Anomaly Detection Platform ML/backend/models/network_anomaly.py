import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def preprocess_network_data(df: pd.DataFrame):
    # Encode categorical columns if any
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def train_network_model(df: pd.DataFrame):
    df_processed = preprocess_network_data(df)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df_processed)
    return model

def predict_network_anomaly(model, df: pd.DataFrame):
    df_processed = preprocess_network_data(df)
    preds = model.predict(df_processed)
    scores = model.decision_function(df_processed)
    # -1 indicates anomaly, 1 indicates normal
    anomaly_indices = [i for i, pred in enumerate(preds) if pred == -1]
    total = int(len(preds))
    anomalies = len(anomaly_indices)
    is_anomaly = [pred == -1 for pred in preds]
    return {
        "Point Anomalies": [f"Anomaly at row {i} (score: {scores[i]:.3f})" for i in anomaly_indices],
        "Contextual Anomalies": [],
        "Collective Anomalies": [],
        "Novelty Detection": [],
        "scores": scores.tolist(),
        "is_anomaly": is_anomaly
    }
