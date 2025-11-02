from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Plant.id API key
API_KEY = 'UhNCFvsUe1PE4yZ0vEsoEIE7Pb5nMrf8EeJiQBqeRKonHrwyOE'

# Fallback to local model if API fails
try:
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import io
    model = tf.keras.models.load_model('plant_disease_model.h5')
    class_indices = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20, 'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37}
    class_names = {v: k for k, v in class_indices.items()}
    USE_LOCAL_MODEL = True
except ImportError:
    USE_LOCAL_MODEL = False

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read image and encode to base64
        image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Plant.id API request for identification
        url_identify = 'https://api.plant.id/v2/identify'
        headers = {
            'Api-Key': API_KEY,
            'Content-Type': 'application/json'
        }
        data_identify = {
            'images': [encoded_image],
            'modifiers': ['crops_fast', 'similar_images'],
            'plant_details': ['common_names', 'url', 'description', 'taxonomy', 'rank', 'gbif_id']
        }

        response_identify = requests.post(url_identify, json=data_identify, headers=headers)
        if response_identify.status_code != 200:
            return jsonify({'error': 'Identification API request failed', 'details': response_identify.text}), response_identify.status_code

        result_identify = response_identify.json()
        if not result_identify['suggestions']:
            return jsonify({'error': 'No plant identified'}), 404

        top_suggestion = result_identify['suggestions'][0]
        plant_name = top_suggestion['plant_name']
        probability = top_suggestion['probability']
        common_names = top_suggestion['plant_details']['common_names']

        # Plant.id API request for health assessment
        url_health = 'https://api.plant.id/v2/health_assessment'
        data_health = {
            'images': [encoded_image],
            'modifiers': ['crops_fast', 'similar_images'],
            'disease_details': ['common_names', 'url', 'description']
        }

        response_health = requests.post(url_health, json=data_health, headers=headers)
        health_status = 'Unknown'
        health_probability = 0.0
        if response_health.status_code == 200:
            result_health = response_health.json()
            diseases = result_health['health_assessment']['diseases']
            if diseases:
                top_disease = diseases[0]
                if top_disease['probability'] > 0.3:  # Threshold to consider it diseased
                    health_status = top_disease['name']
                    health_probability = top_disease['probability']
                else:
                    health_status = 'Healthy'
                    health_probability = result_health['health_assessment']['is_healthy_probability']
            else:
                health_status = 'Healthy'
                health_probability = result_health['health_assessment']['is_healthy_probability']
        else:
            # If health API fails, fallback to local model if available
            if USE_LOCAL_MODEL:
                # Use local model for disease detection
                image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
                image = image.resize((150, 150))
                image = image.convert('RGB')
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                predictions = model.predict(image_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                predicted_class = class_names[predicted_class_idx]
                health_status = predicted_class
                health_probability = confidence

        return jsonify({
            'plant_name': plant_name,
            'probability': probability,
            'common_names': common_names,
            'health_status': health_status,
            'health_probability': health_probability
        })

directory = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory(directory, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(debug=True)
