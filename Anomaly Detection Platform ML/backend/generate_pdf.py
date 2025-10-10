from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def generate_anomaly_report_pdf(anomaly_type, result_data, anomalies):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 100, f"Anomaly Detection Report - {anomaly_type.capitalize()}")

    y = height - 150
    c.drawString(100, y, "Detected Anomalies:")
    y -= 20

    for anomaly in anomalies:
        c.drawString(120, y, f"{anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        y -= 20
        if y < 100:
            c.showPage()
            y = height - 100

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Sample code (can be removed)
# c = canvas.Canvas("sample.pdf")
# c.drawString(100, 750, "This is a sample PDF for testing anomaly detection.")
# c.save()
