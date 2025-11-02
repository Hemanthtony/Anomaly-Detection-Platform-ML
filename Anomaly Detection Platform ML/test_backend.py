oiimport requests
import json
import time
import os

BASE_URL = "http://localhost:8003"

def test_predict(anomaly_type, filename, expected_keys=None):
    """Test prediction endpoint for given type and file."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping.")
        return False
    
    url = f"{BASE_URL}/predict/{anomaly_type}"
    with open(filename, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"{anomaly_type.capitalize()} predict response for {filename}: Status {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
        if expected_keys:
            for key in expected_keys:
                assert key in result['result'], f"Missing key '{key}' in {anomaly_type} response"
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_train(anomaly_type, filename):
    """Test training endpoint (basic, assumes no model exists)."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping.")
        return False
    
    url = f"{BASE_URL}/train/{anomaly_type}"
    with open(filename, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"{anomaly_type.capitalize()} train response for {filename}: Status {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_realtime(anomaly_type):
    """Test realtime detection endpoint."""
    if anomaly_type == "network":
        url = f"{BASE_URL}/detect/network/realtime"
        data = {"features": [0.1, 0.2, 0.3]}  # Sample features
    elif anomaly_type == "weather":
        url = f"{BASE_URL}/detect/weather/realtime"
        data = {"features": [25.0, 60.0, 1013.0]}  # Sample weather features
    elif anomaly_type == "spam":
        url = f"{BASE_URL}/detect/spam/realtime"
        data = {"text": "Sample spam text"}
    else:
        return False
    
    response = requests.post(url, json=data)
    print(f"{anomaly_type.capitalize()} realtime response: Status {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
        assert "is_anomaly" in result
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_generate_pdf():
    """Test PDF generation endpoint."""
    url = f"{BASE_URL}/generate/pdf"
    data = {
        "anomalyType": "spam",
        "resultData": {"sample": "data"},
        "anomalies": [{"type": "Point Anomaly", "description": "Test"}]
    }
    response = requests.post(url, json=data)
    print(f"PDF generation response: Status {response.status_code}")
    if response.status_code == 200:
        print("PDF generated successfully (content-type: application/pdf)")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_websocket(anomaly_type, duration=5):
    """Simple WebSocket test using requests (limited), or note for manual."""
    # For full WebSocket, recommend websocket-client; here, just check connection via logs
    print(f"WebSocket test for {anomaly_type}: Manual connection to ws://{BASE_URL.replace('http', 'ws')}/ws/anomaly/{anomaly_type}")
    print("Check server logs for connection acceptance and no disconnect errors.")
    # Simulate by noting active terminal output
    time.sleep(duration)  # Wait to observe logs
    return True  # Assume pass if no immediate error

# Run tests
print("Starting thorough backend tests...")

# 1. Prediction tests (CSV)
test_predict("network", "test_network_correct.csv", ["Point Anomalies", "scores", "is_anomaly"])
test_predict("weather", "sample_weather.csv", ["Point Anomalies"])
test_predict("spam", "sample_spam.csv", ["Point Anomalies", "scores", "is_anomaly"])  # Verify fix

# 2. PDF prediction for spam (critical fix)
test_predict("spam", "test.pdf", ["Point Anomalies", "scores", "is_anomaly"])

# 3. Training tests (if models not loaded)
test_train("spam", "sample_spam.csv")

# 4. Realtime endpoints
test_realtime("network")
test_realtime("weather")
test_realtime("spam")

# 5. Other endpoints
test_generate_pdf()
test_predict("pdf", "test.pdf")  # /detect/pdf

# 6. WebSocket (observe logs)
test_websocket("spam", 3)

print("Thorough testing completed. Review outputs for any failures.")
