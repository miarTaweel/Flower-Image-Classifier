import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

IMAGE_SIZE = 224

model = tf.keras.models.load_model('flower_classifier.h5')

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

def predict(image, top_k=5):
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    top_k_probs, top_k_indices = tf.math.top_k(predictions[0], k=top_k)
    top_k_probs = top_k_probs.numpy().tolist()
    top_k_classes = [str(idx.numpy()) for idx in top_k_indices]
    flower_names = [class_names[c] for c in top_k_classes]
    return top_k_probs, flower_names

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image_array = np.asarray(image)
    top_k = int(request.form.get('top_k', 5))
    probs, names = predict(image_array, top_k)
    results = [{'name': name, 'probability': round(prob * 100, 2)} for name, prob in zip(names, probs)]
    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True)
