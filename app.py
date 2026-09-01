"""
Fruit Quality Detection - Flask backend
Reconstructed from the mini-project report's implementation section
(load model -> preprocess image -> predict -> classify good/bad -> display).
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 100

# --- Load the trained model and define class labels (Figure 5.1 in report) ---
MODEL_PATH = os.path.join("model", "fruit_quality_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# NOTE: These class labels are taken directly from the report (Fig. 5.1).
# Update this list if your actual training labels differ.
class_labels = [
    "Good Orange", "Bad Orange",
    "Good Apple", "Bad Apple",
    "Good Pomegranate", "Bad Pomegranate",
]


def predict_fruit(image_path):
    """Preprocess and predict image (Figure 5.2 in report)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Determine if the fruit is good or bad (Figure 5.3 in report)
    if "Bad" in class_labels[predicted_class]:
        fruit_quality = "Bad"
    else:
        fruit_quality = "Good"

    return fruit_quality, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            fruit_quality, conf = predict_fruit(filepath)
            if fruit_quality is None:
                prediction_text = "Error: Unable to load image."
            else:
                prediction_text = fruit_quality
                confidence = round(conf, 2)

            image_url = filepath

    return render_template(
        "index.html",
        prediction=prediction_text,
        confidence=confidence,
        image_url=image_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
