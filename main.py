from fastapi import FastAPI, File, UploadFile
import uvicorn
import io
from PIL import Image
import tensorflow as tf
import numpy as np
import gdown
import os

app = FastAPI()

# Google Drive model URL (replace with your actual shareable link)
MODEL_URL = "https://drive.google.com/uc?id=1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"  # Replace with actual ID
MODEL_PATH = "skin_disease_model.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["akiec", "bcc", "df", "mel", "nv", "vasc"]  # Adjust according to your model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {"prediction": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
