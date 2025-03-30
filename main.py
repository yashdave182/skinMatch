import os
import gdown
import tensorflow as tf

# Disable GPU to prevent errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Reduce TensorFlow logging
tf.get_logger().setLevel("ERROR")

# Define the Google Drive file ID and local model path
MODEL_ID = "1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"  # Replace with actual ID
MODEL_PATH = "model.h5"

# Check if model file exists, otherwise download
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
else:
    print("Model file already exists.")

# Load the model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
else:
    print("Model file not found! Ensure the download was successful.")

# Print available devices
print("Available devices:", tf.config.list_physical_devices())
