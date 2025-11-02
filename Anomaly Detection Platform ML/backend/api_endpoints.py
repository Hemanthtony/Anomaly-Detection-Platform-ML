from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io

app = FastAPI()

# Sample data models for anomaly detection requests and responses
class NetworkData(BaseModel):
    features: List[float]

class WeatherData(BaseModel):
    features: List[float]

class SpamData(BaseModel):
    text: str

class AnomalyResult(BaseModel):
    is_anomaly: bool
    score: Optional[float] = None
    details: Optional[str] = None

class PDFResult(BaseModel):
    anomalies: List[dict]

@app.post("/detect/network", response_model=AnomalyResult)
async def detect_network_anomaly(data: NetworkData):
    # Placeholder logic for network anomaly detection
    # Replace with actual model inference
    if sum(data.features) > 10:
        return AnomalyResult(is_anomaly=True, score=0.95, details="High network anomaly score")
    return AnomalyResult(is_anomaly=False, score=0.1)

@app.post("/detect/weather", response_model=AnomalyResult)
async def detect_weather_anomaly(data: WeatherData):
    # Placeholder logic for weather anomaly detection
    if sum(data.features) < 5:
        return AnomalyResult(is_anomaly=True, score=0.85, details="Low weather anomaly score")
    return AnomalyResult(is_anomaly=False, score=0.2)

@app.post("/detect/spam", response_model=AnomalyResult)
async def detect_spam(data: SpamData):
    # Placeholder logic for spam detection
    if "spam" in data.text.lower():
        return AnomalyResult(is_anomaly=True, score=0.99, details="Spam detected")
    return AnomalyResult(is_anomaly=False, score=0.05)

@app.post("/detect/pdf", response_model=PDFResult)
async def detect_pdf_anomaly(file: UploadFile = File(...)):
    # Placeholder logic for PDF anomaly detection
    # In a real implementation, you would process the PDF file here
    # For now, return mock anomaly data with varying counts
    import random
    types = ["Point Anomaly", "Contextual Anomaly", "Collective Anomaly", "Novelty Detection"]
    anomalies = []
    for _ in range(random.randint(4, 10)):
        anomaly_type = random.choice(types)
        descriptions = {
            "Point Anomaly": ["Spike in data at page {}".format(random.randint(1, 20)), "Outlier value detected"],
            "Contextual Anomaly": ["Unusual pattern in section {}".format(random.randint(1, 10)), "Context mismatch"],
            "Collective Anomaly": ["Group behavior anomaly in table {}".format(random.randint(1, 5)), "Collective deviation"],
            "Novelty Detection": ["New pattern detected in graph", "Unseen data point"]
        }
        desc = random.choice(descriptions[anomaly_type])
        anomalies.append({"type": anomaly_type, "description": desc})
    return PDFResult(anomalies=anomalies)

@app.post("/predict/network")
async def predict_network_anomaly(file: UploadFile = File(...)):
    # Handle file upload for network anomaly detection
    if file.filename.endswith('.csv'):
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        features = df.select_dtypes(include=[float, int]).values.flatten().tolist()
        # Use the model from app.py models dict
        from .app import models, predict_network_anomaly as predict_func
        model = models.get("network")
        if model is None:
            raise HTTPException(status_code=400, detail="Network model not trained yet")
        result = predict_func(model, df)
        return {"result": result}
    else:
        raise HTTPException(status_code=400, detail="Only CSV files are supported for network anomaly detection")

@app.post("/predict/spam")
async def predict_spam(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    elif file.filename.endswith('.pdf'):
        contents = await file.read()
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        df = pd.DataFrame({'text': lines})
    else:
        raise HTTPException(status_code=400, detail="Only CSV or PDF files are supported for spam detection")
    from .app import models, predict_spam as predict_func
    model = models.get("spam")
    if model is None:
        raise HTTPException(status_code=400, detail="Spam model not trained yet")
    result = predict_func(model, df)
    return {"result": result}

@app.post("/predict/weather")
async def predict_weather(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        from .app import models, predict_weather_anomaly as predict_func
        model = models.get("weather")
        if model is None:
            raise HTTPException(status_code=400, detail="Weather model not trained yet")
        result = predict_func(model, df)
        return {"result": result}
    else:
        raise HTTPException(status_code=400, detail="Only CSV files are supported for weather anomaly detection")
