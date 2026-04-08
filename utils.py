import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    top_k_probs, top_k_indices = tf.math.top_k(predictions[0], k=top_k)
    top_k_probs = top_k_probs.numpy().tolist()
    top_k_classes = [str(idx.numpy()) for idx in top_k_indices]
    return top_k_probs, top_k_classes
