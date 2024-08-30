import io
import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
import pytesseract
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai


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
migrate = Migrate(app, db)

class Ingredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ingredient = db.Column(db.String(255), nullable=False, unique=True)
    risk = db.Column(db.Integer, nullable=False)

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


def calculate_risk(ingredients: list[str]) -> dict:
    """
    Calculates the risk of a product based on the ingredients and their risks in the database.

    :param ingredients: List of ingredients in the product.
    :return: A dictionary with the normalized risk, the risk classification, and the list of risky ingredients.
    """
    n_total = len(ingredients)
    total_risks = 0
    n_risky = 0
    risky_ingredients = []

    for ingredient in ingredients:
        ingredient_upper = ingredient.upper().strip()
        ing = Ingredient.query.filter_by(ingredient=ingredient_upper).first()
        if ing:
            total_risks += ing.risk
            n_risky += 1
            if ingredient_upper not in risky_ingredients:
                risky_ingredients.append(ingredient_upper.title())

    if n_total == 0:
        return {
            'normalized_risk': 0,
            'classification': "N/A",
            'risky_ingredients': []
        }

    total_risk = total_risks / n_total
    normalized_risk = total_risk * 100

    if normalized_risk <= 33:
        classification = "Bajo riesgo"
    elif normalized_risk <= 66:
        classification = "Mediano riesgo"
    else:
        classification = "Alto riesgo"

    return {
        'normalized_risk': round(normalized_risk, 2),
        'classification': classification,
        'risky_ingredients': risky_ingredients
    }


def normalize_ingredient(ingredient: str) -> str:
    """
    Normalize the ingredient name by converting to uppercase and removing extra spaces.
    """
    return ' '.join(ingredient.upper().split())

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.xlsx'):
        try:
            df = pd.read_excel(file)

            if 'ingredient' not in df.columns or 'risk' not in df.columns:
                return jsonify({'error': 'Invalid Excel format'}), 400

            df['ingredient'] = df['ingredient'].apply(normalize_ingredient)
            existing_ingredients = {i.ingredient: i for i in Ingredient.query.all()}

            new_ingredients = []
            for index, row in df.iterrows():
                ingredient = row['ingredient']
                risk = row['risk']

                if ingredient in existing_ingredients:
                    existing_ingredients[ingredient].risk = risk
                else:
                    new_ingredients.append(Ingredient(ingredient=ingredient, risk=risk))

            if new_ingredients:
                db.session.add_all(new_ingredients)

            db.session.commit()
            return jsonify({'message': 'Ingredients uploaded successfully'}), 200

        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload an image and extract the ingredients and calculate the risk of the product.
    :return: JSON response with the risk of the product and the list of risky ingredients.
    {
    "classification": "Mediano riesgo",
    "normalized_risk": 54.29,
    "risky_ingredients": [
        "Ácido Palmítico",
        "Ácido Esteárico",
        "Ácido Málico",
        "Ácido Cítrico"
    ]
}
    """
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
            risk = calculate_risk(ingredients.split(','))
            #logging.debug(f'Ingredients: {ingredients}')
            return jsonify(
                risk
            ), 200
        except Exception as e:
            print(e)
            return jsonify({
                'error': 'Ocurrió un error al procesar la imagen'
            }), 500

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
