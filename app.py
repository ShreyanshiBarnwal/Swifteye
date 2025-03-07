from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

app = Flask(__name__)



# Google Drive file ID (replace with your actual file ID)
FILE_ID = "1ZO02nwFkmAgiL6x0DSyW4N19bXf9MjOA"
MODEL_PATH = "model.h5"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/file/d/1ZO02nwFkmAgiL6x0DSyW4N19bXf9MjOA/view?usp=sharing", MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file)
    image = preprocess_image(image)

    prediction = model.predict(image)
    result = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
