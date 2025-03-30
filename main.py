from flask import Flask, request, jsonify
import io
from PIL import Image
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React Native

# Load trained model
MODEL_PATH = "skin_disease_model.h5"
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
