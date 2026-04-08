import argparse
import json
import tensorflow as tf
from utils import predict

parser = argparse.ArgumentParser(description='Predict flower name from an image')
parser.add_argument('image_path', type=str, help='Path to the image file')
parser.add_argument('saved_model', type=str, help='Path to the saved Keras model')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to JSON file mapping labels to flower names')

args = parser.parse_args()

model = tf.keras.models.load_model(args.saved_model)

with open(args.category_names, 'r') as f:
    class_names = json.load(f)

probs, classes = predict(args.image_path, model, args.top_k)
flower_names = [class_names[c] for c in classes]

print('\nPrediction Results:')
print('-' * 30)
for i in range(len(probs)):
    print(f'{i+1}. {flower_names[i]}: {probs[i]*100:.2f}%')
