import os
import gdown
import uvicorn
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# 🔹 Load model from Google Drive if not available
MODEL_URL = "https://drive.google.com/uc?id=1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# 🔹 Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Define Class Labels
CLASS_LABELS = {
    0: "Actinic keratoses (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

# 🔹 Prediction API Endpoint
@app.post("/predict")
async def predict(file: bytes):
    try:
        # Load image
        image = Image.open(BytesIO(file)).resize((224, 224))  # Resize for model
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return {
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

# 🔹 Run API (Uses Render's PORT)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render auto-assigns a port
    uvicorn.run(app, host="0.0.0.0", port=port)
