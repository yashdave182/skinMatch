import os
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import uvicorn
import gdown

# Disable GPU usage on Render and TensorRT warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

app = FastAPI()

# Check TensorFlow status
print("Using TensorFlow version:", tf.__version__)
print("Devices available:", tf.config.list_physical_devices())

# Define model path and download if not available
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/file/d/1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3/view?usp=drive_link"  # Replace with actual model link
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the TensorFlow/Keras model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return {"prediction": int(predicted_class), "confidence": float(np.max(predictions))}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
