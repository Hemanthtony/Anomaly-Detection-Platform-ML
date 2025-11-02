# Anomaly Detection Platform ML

A comprehensive web-based platform for detecting anomalies in various types of data using advanced machine learning algorithms. The platform supports real-time anomaly detection, model training, and PDF report generation for network traffic, weather patterns, and spam messages.

## Features

- **Multi-Type Anomaly Detection**: Supports detection of Point Anomalies, Contextual Anomalies, Collective Anomalies, and Novelty Detection
- **Real-Time Detection**: WebSocket-based real-time anomaly monitoring
- **Model Training**: Train custom models for network, weather, and spam detection
- **File Upload Support**: Accepts CSV, PDF, and ZIP files for training and prediction
- **PDF Report Generation**: Generate detailed anomaly reports in PDF format
- **MongoDB Integration**: Persistent model storage using MongoDB GridFS
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Web Interface**: User-friendly HTML frontend for easy interaction

## Supported Anomaly Types

### Network Anomalies
- Detect unusual patterns in network traffic data
- Supports features like packet counts, bandwidth usage, etc.

### Weather Anomalies
- Identify anomalous weather patterns
- Analyzes temperature, humidity, and pressure data

### Spam Detection
- Classify messages as spam or legitimate
- Text-based anomaly detection using NLP techniques

## Installation

### Prerequisites
- Python 3.8+
- MongoDB (for model storage)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd anomaly-detection-platform-ml
   ```

2. **Create and activate virtual environment**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB**
   - Ensure MongoDB is running on your system
   - Update connection settings in `mongo_storage.py` if needed

5. **Run the application**
   ```bash
   python app.py
   ```

The application will start on `http://localhost:8003`

## Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8003`
2. Click "Get Started" to access the login page
3. Use the provided HTML interfaces for different anomaly detection tasks

### API Usage

#### Train a Model
```bash
curl -X POST "http://localhost:8003/train/{anomaly_type}" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"
```

Replace `{anomaly_type}` with `network`, `weather`, or `spam`

#### Make Predictions
```bash
curl -X POST "http://localhost:8003/predict/{anomaly_type}" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_data.csv"
```

#### Real-Time Detection
```bash
curl -X POST "http://localhost:8003/detect/{anomaly_type}/realtime" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3]}'
```

## API Endpoints

### Training Endpoints
- `POST /train/{anomaly_type}` - Train a model for specified anomaly type
- Supports file uploads (CSV, PDF, ZIP)

### Prediction Endpoints
- `POST /predict/{anomaly_type}` - Predict anomalies in uploaded data
- `POST /detect/{anomaly_type}/realtime` - Real-time anomaly detection

### WebSocket Endpoints
- `WS /ws/anomaly/{anomaly_type}` - Real-time anomaly detection via WebSocket

### Utility Endpoints
- `POST /generate/pdf` - Generate PDF reports
- `POST /detect/pdf` - PDF anomaly detection (placeholder)

### Static Files
- `GET /` - Serves the main HTML interface
- Static files served from root directory

## Technologies Used

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **Joblib**: Model serialization
- **PyMongo**: MongoDB driver
- **PyPDF2**: PDF text extraction
- **Tabula-py**: PDF table extraction
- **ReportLab**: PDF generation

### Frontend
- **HTML5/CSS3**: Responsive web interface
- **JavaScript**: Client-side interactions
- **Feather Icons**: Icon library

### Database
- **MongoDB**: Document database for model storage
- **GridFS**: File storage system for large model files

## Project Structure

```
anomaly-detection-platform-ml/
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   ├── network_anomaly.py # Network anomaly detection logic
│   │   ├── weather_anomaly.py # Weather anomaly detection logic
│   │   └── spam_detection.py  # Spam detection logic
│   ├── mongo_storage.py       # MongoDB integration
│   ├── generate_pdf.py        # PDF report generation
│   └── ...
├── index.html                 # Main landing page
├── login.html                 # Login interface
├── select_anomaly.html        # Anomaly type selection
├── network_anomaly.html       # Network anomaly interface
├── weather_anomaly.html       # Weather anomaly interface
├── spam_detection.html        # Spam detection interface
├── anomaly_results.html       # Results display
├── firebase-config.js         # Firebase configuration
└── README.md                  # This file
```

## Data Formats

### Network Data
CSV with columns for network features (e.g., packet_count, bandwidth, latency)

### Weather Data
CSV with columns: temperature, humidity, pressure (minimum 3 columns)

### Spam Data
CSV with 'text' column containing message content

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

**Hemanth**  
Email: hh8867708@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository or contact the development team.

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Dashboard with analytics and visualizations
- [ ] Support for additional anomaly types
- [ ] Model performance monitoring
- [ ] API rate limiting and security features
- [ ] Containerization with Docker
- [ ] Cloud deployment options
