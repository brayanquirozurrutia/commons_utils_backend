from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
import re

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return Image.fromarray(image)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)
            text = pytesseract.image_to_string(processed_image, lang='spa', config='--psm 6')
            ingredients = extract_ingredients(text)
            return jsonify({'text': text, 'ingredients': ingredients}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp'}

def extract_ingredients(text):
    pattern = re.compile(r'\b(?:ingredientes|ingrediente|contiene)\b[:\s]*(.*)', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).split(',')
    return []

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
