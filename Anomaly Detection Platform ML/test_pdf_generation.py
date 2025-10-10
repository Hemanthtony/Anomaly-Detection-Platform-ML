import requests

url = "http://localhost:8003/generate/pdf"
payload = {
    "anomalyType": "network",
    "resultData": {},
    "anomalies": [
        {"type": "Point Anomaly", "description": "Test anomaly"}
    ]
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("test_report.pdf", "wb") as f:
        f.write(response.content)
    print("PDF report downloaded successfully as test_report.pdf")
else:
    print(f"Failed to generate PDF report. Status code: {response.status_code}")
    print("Response:", response.text)
