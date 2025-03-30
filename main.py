from flask import Flask, request, jsonify
import io
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from flask_cors import CORS
import gdown
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React Native

# Google Drive file ID extracted from:
# https://drive.google.com/file/d/1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3/view?usp=sharing
DRIVE_FILE_ID = "1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"
MODEL_PATH = "skin_disease_model.h5"

def download_model():
    """Downloads the model from Google Drive if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")

# Download and load model
try:
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Adjust according to your model

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Open and process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure RGB format
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions[0])]  # Index into first prediction
        confidence = float(np.max(predictions[0]))  # Index into first prediction
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
