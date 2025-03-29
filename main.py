import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# Google Drive File ID (Extracted from the link)
FILE_ID = "1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"

# Download the model from Google Drive
def download_model(file_id, filename="skin_disease_model.h5"):
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(drive_url, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Model downloaded successfully: {filename}")
        return filename
    else:
        print("❌ Error downloading model.")
        return None

# Load the model
def load_model():
    model_filename = "skin_disease_model.h5"
    
    # Check if the model exists locally; if not, download it
    if not os.path.exists(model_filename):
        print("🔄 Downloading model...")
        model_filename = download_model(FILE_ID, model_filename)
    
    if model_filename:
        model = tf.keras.models.load_model(model_filename)
        print("✅ Model loaded successfully!")
        return model
    else:
        print("❌ Failed to load the model.")
        return None

# Preprocess image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize to model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Make prediction
def predict(image_path, model):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get class index
    confidence = np.max(prediction)  # Get confidence score
    
    # Class labels (Change these as per your model)
    class_labels = ["Eczema", "Psoriasis", "Melanoma", "Acne", "Healthy Skin"]

    print(f"🔍 Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")
    return class_labels[predicted_class], confidence

# Run the program
if __name__ == "__main__":
    model = load_model()
    
    if model:
        image_path = input("Enter the image file path: ")
        if os.path.exists(image_path):
            disease, conf = predict(image_path, model)
            print(f"🔹 Predicted Skin Condition: {disease} ({conf * 100:.2f}% confidence)")
        else:
            print("❌ Image file not found!")
