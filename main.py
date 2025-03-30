import os
import gdown
import tensorflow as tf

# Force TensorFlow to use only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress unnecessary TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# Google Drive model ID and file path
MODEL_ID = "1t4hK_d1N8a2nTl-9ZAiXuGb6T8rEcKW3"  # Replace with actual ID
MODEL_PATH = "model.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
        print("Download completed.")
    except Exception as e:
        print("Error downloading model:", e)
        exit(1)
else:
    print("Model file already exists.")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit(1)

# Display available devices
devices = tf.config.list_physical_devices()
print("📌 Available devices:", devices)

# Check if TensorFlow is using the GPU (it shouldn't)
if any("GPU" in str(device) for device in devices):
    print("⚠ Warning: GPU is still being used despite disabling it.")
else:
    print("✅ TensorFlow is correctly set to CPU-only mode.")
