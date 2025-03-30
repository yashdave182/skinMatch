from flask import Flask, request, jsonify
import io
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React Native

# Google Drive file ID (Extract from your shared link)
DRIVE_FILE_ID = "1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"


def download_model():
    """Downloads the model from Google Drive if it is not present locally."""
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            print("Failed to download model.")
            raise Exception("Failed to download model from Google Drive.")

# Download model before loading
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Adjust according to your model

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return jsonify({"prediction": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
