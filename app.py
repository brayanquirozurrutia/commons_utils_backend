from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=OPENAI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

db = SQLAlchemy(app)

class Ingredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ingredient = db.Column(db.String(255), nullable=False)
    risk = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<Ingredient {self.ingredient}>"

with app.app_context():
    db.create_all()


def preprocess_image(image: Image) -> Image:
    """
    Preprocess the image to improve the OCR results
    :param image: Image to preprocess
    :return: Preprocessed image
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return Image.fromarray(image)

def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed
    :param filename: Filename to check
    :return: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp'}

def process_text(text: str) -> str:
    """
    Process the text to extract the ingredients. the function looks for the word 'ingredientes' and extracts the text
    after that word.
    :param text: Text to process
    :return: Ingredients extracted from the text
    """
    normalized_text = text.lower()

    start_index = normalized_text.find('ingredientes:')
    if start_index == -1:
        for variation in [
            'ingrediente:',
            'ingredientes',
            'ingredientes:',
            'ingredientes.',
            'ingredientes...',
        ]:
            start_index = normalized_text.find(variation)
            if start_index != -1:
                break

    if start_index == -1:
        return ''

    start_index += len('ingredientes:')

    return text[start_index:].strip()

def extract_ingredients(ingredients: str) -> str:
    """
    Extract the ingredients from the text using the Gemini model
    :param ingredients: Ingredients to extract
    :return: Extracted ingredients
    """
    response = model.generate_content(f"""Identify all the ingredients present in this text and return them separated
    by a comma. For example:
    MILK, COCOA POWDER, SALT, PALM OIL...
    The ingredients are in Spanish and you must return them in Spanish. The text is:
    {ingredients}""")
    return response.text


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
            #logging.debug(f'Text: {text}')
            normalized_text = process_text(text)
            #logging.debug(f'Normalized text: {normalized_text}')
            ingredients = extract_ingredients(normalized_text)
            #logging.debug(f'Ingredients: {ingredients}')
            return jsonify({'ingredients': ingredients}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
