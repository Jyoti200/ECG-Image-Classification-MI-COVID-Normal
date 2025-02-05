import os
import logging
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model safely
MODEL_PATH = r"C:\Users\91954\Downloads\ECG_dataset\ECG_dataset.keras"

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        logging.info("Model loaded successfully.")
    else:
        logging.error(f"Model file not found at {MODEL_PATH}.")
        model = None  # Handle missing model gracefully
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Class labels
CLASSES = ['Normal', 'Covid-19', 'MI', 'MI_History', 'Abnormal Heartbeat']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(image):
    try:
        # Convert image to RGB (in case it's grayscale or has an alpha channel)
        image = image.convert("RGB")
        image = image.resize((128, 128))  # Resize to match model input size
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

# Homepage route (Upload form + Display result)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', prediction="No selected file.")

        if not allowed_file(file.filename):
            return render_template('index.html', prediction="Invalid file format. Only PNG, JPG, and JPEG are allowed.")

        if model is None:
            return render_template('index.html', prediction="Model not loaded. Please contact support.")

        try:
            # Open the image
            img = Image.open(file)
            processed_img = preprocess_image(img)

            if processed_img is None:
                return render_template('index.html', prediction="Error processing image.")

            # Make a prediction
            prediction = model.predict(processed_img)
            predicted_index = np.argmax(prediction[0])

            # Ensure the predicted index is within range
            if predicted_index >= len(CLASSES):
                logging.error(f"Invalid prediction index: {predicted_index}")
                return render_template('index.html', prediction="Prediction error.")

            result = CLASSES[predicted_index]
            confidence = round(float(np.max(prediction[0])) * 100, 2)

            return render_template('index.html', prediction=f"{result} ({confidence}% confidence)")

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return render_template('index.html', prediction="Error making prediction.")

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
