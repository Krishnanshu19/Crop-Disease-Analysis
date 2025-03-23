from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Define class labels
class_labels = ["blight", "spot", "healthy", "rust"]

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    print("Request received!")  # Debugging log
    print("Request files:", request.files)  # Log uploaded files

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))  # Open image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image) 
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
