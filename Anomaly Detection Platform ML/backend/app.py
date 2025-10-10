import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
from io import StringIO
from typing import Dict, List, Optional
import os
import joblib
import io
import json
import numpy as np
from .mongo_storage import save_model_file, get_model_file

from models.network_anomaly import train_network_model, predict_network_anomaly
from models.weather_anomaly import train_weather_model, predict_weather_anomaly
from models.spam_detection import train_spam_model, predict_spam

from .generate_pdf import generate_anomaly_report_pdf

from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.post("/detect/pdf")
async def detect_pdf_anomaly(file: UploadFile = File(...)):
    # Placeholder logic for PDF anomaly detection
    # In a real implementation, you would process the PDF file here
    # For now, return mock anomaly data
    anomalies = [
        {"type": "Point Anomaly", "description": "Spike in data at page 5"},
        {"type": "Contextual Anomaly", "description": "Unusual pattern in section 3"},
        {"type": "Collective Anomaly", "description": "Group behavior anomaly in table 2"},
        {"type": "Novelty Detection", "description": "New pattern detected in graph"}
    ]
    return {"anomalies": anomalies}

# Allow CORS for frontend running on different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models for real-time detection
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

# Load models from local saved_models directory only
models: Dict[str, object] = {}
for anomaly_type in ["network", "weather", "spam"]:
    try:
        model_path = os.path.join(MODEL_DIR, f"{anomaly_type}_model.joblib")
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            # Fix for spam model being a tuple instead of model object
            if anomaly_type == "spam" and isinstance(loaded_model, tuple):
                # Assume the first element is the actual model
                models[anomaly_type] = loaded_model[0]
                print(f"Loaded spam model corrected from tuple for: {model_path}")
            else:
                models[anomaly_type] = loaded_model
            print(f"Loaded {anomaly_type} model from local: {model_path}")
            if hasattr(models[anomaly_type], 'feature_names_in_'):
                print(f"{anomaly_type} model features: {models[anomaly_type].feature_names_in_}")
        else:
            print(f"No local model found for {anomaly_type}")
            models[anomaly_type] = None
    except Exception as e:
        print(f"Error loading {anomaly_type} model: {e}")
        models[anomaly_type] = None

@app.post("/train/{anomaly_type}")
async def train_model(anomaly_type: str, file: UploadFile = File(...)):
    if anomaly_type not in models:
        raise HTTPException(status_code=400, detail="Invalid anomaly type")

    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
        elif filename.endswith(".pdf"):
            import tempfile
            try:
                import PyPDF2
            except ImportError:
                raise HTTPException(status_code=500, detail="PyPDF2 module not installed. Please install it to process PDF files.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(contents)
                tmp_pdf.flush()
                reader = PyPDF2.PdfReader(tmp_pdf.name)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            if anomaly_type == "spam":
                df = pd.DataFrame({'text': text.split('\n'), 'label': [0] * len(text.split('\n'))})
            else:
                df = pd.read_csv(StringIO(text))
        elif filename.endswith(".zip"):
            import tempfile
            import zipfile
            with tempfile.TemporaryDirectory() as tmpdirname:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(contents)
                    tmp_zip.flush()
                    with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                        zip_ref.extractall(tmpdirname)
                    import os
                    csv_found = False
                    for root, dirs, files in os.walk(tmpdirname):
                        for file_in_zip in files:
                            if file_in_zip.endswith(".csv"):
                                csv_path = os.path.join(root, file_in_zip)
                                df = pd.read_csv(csv_path)
                                csv_found = True
                                break
                        if csv_found:
                            break
                    if not csv_found:
                        raise HTTPException(status_code=400, detail="No CSV file found in ZIP archive")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, PDF, or ZIP.")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {str(e)}")
        error_msg = str(e) or "Unknown error occurred while reading the file."
        raise HTTPException(status_code=400, detail=f"Error reading file: {error_msg}")

    if anomaly_type == "network":
        model = train_network_model(df)
    elif anomaly_type == "weather":
        model = train_weather_model(df)
    elif anomaly_type == "spam":
        model = train_spam_model(df)
    else:
        raise HTTPException(status_code=400, detail="Unsupported anomaly type")

    models[anomaly_type] = model
    # Save model to MongoDB GridFS
    import io
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_bytes = model_buffer.getvalue()
    save_model_file(model_bytes, f"{anomaly_type}_model.joblib")
    return {"message": f"{anomaly_type.capitalize()} model trained and saved successfully"}

@app.post("/predict/{anomaly_type}")
async def predict_anomaly(anomaly_type: str, file: UploadFile = File(...)):
    if anomaly_type not in models or models[anomaly_type] is None:
        raise HTTPException(status_code=400, detail=f"{anomaly_type.capitalize()} model not trained yet")

    contents = await file.read()
    filename = file.filename.lower()

    try:
        if anomaly_type == "weather":
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(StringIO(contents.decode(enc)), header=None, on_bad_lines='skip', engine='python')
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            if df is None:
                raise HTTPException(status_code=400, detail="Unable to decode the file. Please ensure it is a valid CSV file with text encoding.")
            if len(df.columns) >= 3:
                df.columns = ['temperature', 'humidity', 'pressure'] + list(df.columns[3:])
            else:
                raise HTTPException(status_code=400, detail="Weather data must have at least 3 columns (temperature, humidity, pressure)")
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(StringIO(contents.decode("utf-8")))
            except UnicodeDecodeError:
                df = pd.read_csv(StringIO(contents.decode("latin1")))
        elif filename.endswith(".pdf"):
            import tempfile
            import PyPDF2
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(contents)
                tmp_pdf.flush()
                reader = PyPDF2.PdfReader(tmp_pdf.name)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            df = pd.read_csv(StringIO(text))
        elif filename.endswith(".zip"):
            import tempfile
            import zipfile
            with tempfile.TemporaryDirectory() as tmpdirname:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(contents)
                    tmp_zip.flush()
                    with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                        zip_ref.extractall(tmpdirname)
                    import os
                    csv_found = False
                    for root, dirs, files in os.walk(tmpdirname):
                        for file_in_zip in files:
                            if file_in_zip.endswith(".csv"):
                                csv_path = os.path.join(root, file_in_zip)
                                df = pd.read_csv(csv_path)
                                csv_found = True
                                break
                        if csv_found:
                            break
                    if not csv_found:
                        raise HTTPException(status_code=400, detail="No CSV file found in ZIP archive")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, PDF, or ZIP.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    model = models[anomaly_type]

    # Align features with model's expected features
    try:
        expected_features = model.feature_names_in_
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        extra_features = [col for col in df.columns if col not in expected_features]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features in input data: {missing_features}")
        if extra_features:
            # Drop extra columns
            df = df[expected_features]
        else:
            df = df[expected_features]
    except AttributeError:
        # model.feature_names_in_ not available, skip alignment
        pass
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")

    if anomaly_type == "network":
        result = predict_network_anomaly(model, df)
        detailed_result = {
            "Point Anomalies": result.get("point_anomalies", []),
            "Contextual Anomalies": result.get("contextual_anomalies", []),
            "Collective Anomalies": result.get("collective_anomalies", []),
            "Novelty Detection": result.get("novelty_detection", [])
        }
        return {"result": detailed_result}
    elif anomaly_type == "weather":
        result = predict_weather_anomaly(model, df)
        detailed_result = {
            "Point Anomalies": result.get("Point Anomalies", []),
            "Contextual Anomalies": result.get("Contextual Anomalies", []),
            "Collective Anomalies": result.get("Collective Anomalies", []),
            "Novelty Detection": result.get("Novelty Detection", [])
        }
        return {"result": detailed_result}
    elif anomaly_type == "spam":
        result = predict_spam(model, df)
        detailed_result = {
            "Point Anomalies": result.get("point_anomalies", []),
            "Contextual Anomalies": result.get("contextual_anomalies", []),
            "Collective Anomalies": result.get("collective_anomalies", []),
            "Novelty Detection": result.get("novelty_detection", [])
        }
        return {"result": detailed_result}
    else:
        raise HTTPException(status_code=400, detail="Unsupported anomaly type")

@app.websocket("/ws/anomaly/{anomaly_type}")
async def websocket_anomaly(websocket: WebSocket, anomaly_type: str):
    await websocket.accept()
    if anomaly_type not in models or models[anomaly_type] is None:
        await websocket.send_json({"error": f"{anomaly_type.capitalize()} model not trained yet"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            # Use network model for all anomaly types as per initial setup
            if anomaly_type in ["network", "weather"]:
                features = np.array(data["features"]).reshape(1, -1)
            elif anomaly_type == "spam":
                features = np.array([0.5, 0.5, 0.5]).reshape(1, -1)  # Dummy features
            else:
                await websocket.send_json({"error": "Unsupported anomaly type"})
                continue

            prediction = models[anomaly_type].predict(features)
            is_anomaly = bool(prediction[0] == -1)
            # Convert numpy.bool_ to native bool for JSON serialization
            if hasattr(is_anomaly, 'item'):
                is_anomaly = is_anomaly.item()
            score = models[anomaly_type].decision_function(features)[0] if hasattr(models[anomaly_type], 'decision_function') else 0.5
            details = f"Real-time {anomaly_type} anomaly detection"

            result = {
                "is_anomaly": is_anomaly,
                "score": score,
                "details": details
            }
            await websocket.send_json(result)
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
        await websocket.close()

from fastapi.responses import RedirectResponse

# Serve static files from /static directory for frontend HTML pages
import os
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Serving static files from: {static_dir}")
# Mount the directory containing the HTML files and other static assets
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

from fastapi import Request

@app.post("/generate/pdf")
async def generate_pdf_endpoint(request: Request):
    data = await request.json()
    anomaly_type = data.get("anomalyType")
    result_data = data.get("resultData")
    anomalies = data.get("anomalies")
    pdf_bytes = generate_anomaly_report_pdf(anomaly_type, result_data, anomalies)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=anomaly_report_{anomaly_type}.pdf"})

@app.post("/detect/network/realtime", response_model=AnomalyResult)
async def detect_network_realtime(data: NetworkData):
    if "network" not in models or models["network"] is None:
        raise HTTPException(status_code=400, detail="Network model not trained yet")
    import numpy as np
    features = np.array(data.features).reshape(1, -1)
    try:
        prediction = models["network"].predict(features)
        is_anomaly = prediction[0] == -1  # Assuming -1 is anomaly for OneClassSVM
        score = models["network"].decision_function(features)[0] if hasattr(models["network"], 'decision_function') else 0.5
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    return AnomalyResult(is_anomaly=is_anomaly, score=score, details="Real-time network anomaly detection")

@app.post("/detect/weather/realtime", response_model=AnomalyResult)
async def detect_weather_realtime(data: WeatherData):
    if "network" not in models or models["network"] is None:
        raise HTTPException(status_code=400, detail="Network model not trained yet")
    import numpy as np
    features = np.array(data.features).reshape(1, -1)
    prediction = models["network"].predict(features)
    is_anomaly = prediction[0] == -1
    score = models["network"].decision_function(features)[0] if hasattr(models["network"], 'decision_function') else 0.5
    return AnomalyResult(is_anomaly=is_anomaly, score=score, details="Real-time weather anomaly detection")

@app.post("/detect/spam/realtime", response_model=AnomalyResult)
async def detect_spam_realtime(data: SpamData):
    if "network" not in models or models["network"] is None:
        raise HTTPException(status_code=400, detail="Network model not trained yet")
    # Assume text is converted to features somehow, but for now, use network model with dummy features
    import numpy as np
    features = np.array([0.5, 0.5, 0.5]).reshape(1, -1)  # Dummy features
    prediction = models["network"].predict(features)
    is_anomaly = prediction[0] == -1
    score = models["network"].decision_function(features)[0] if hasattr(models["network"], 'decision_function') else 0.5
    return AnomalyResult(is_anomaly=is_anomaly, score=score, details="Real-time spam detection")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
