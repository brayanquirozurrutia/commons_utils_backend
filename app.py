import io
import math
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

    if normalized_risk <= 20:
        classification, classification_message = "Bajo riesgo", "Consumo seguro"
    elif normalized_risk <= 40:
        classification, classification_message = "Medio bajo riesgo", "Consumo moderado"
    elif normalized_risk <= 60:
        classification, classification_message = "Medio riesgo", "Consumo ocasional"
    elif normalized_risk <= 80:
        classification, classification_message = "Medio alto riesgo", "Consumo muy poco frecuente"
    else:
        classification, classification_message = "Alto riesgo", "Consumo no recomendado"

    return {
        'normalized_risk': round(normalized_risk, 2),
        'classification': classification,
        'risky_ingredients': risky_ingredients,
        'classification_message': classification_message
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
            print(e)
            return jsonify({'error': 'Ocurrió un error al procesar el archivo'}), 500

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
    ],
    "classification_message": "Consumo ocasional"
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
            normalized_text = process_text(text)
            ingredients = extract_ingredients(normalized_text)
            risk = calculate_risk(ingredients.split(','))
            return jsonify(
                risk
            ), 200
        except Exception as e:
            print(e)
            return jsonify({
                'error': 'Ocurrió un error al procesar la imagen'
            }), 500

    return jsonify({'error': 'Invalid file format'}), 400

import matplotlib.pyplot as plt
import base64
import itertools


@app.route('/compare_products', methods=['POST'])
def compare_products():
    data = request.json
    combinations = []

    # Generate product combinations
    for r in range(1, len(data) + 1):
        for combo in itertools.combinations(data, r):
            total_cost = sum(item['package_price'] for item in combo)
            total_meters = sum(item['units_per_package'] * item['meters_per_unit'] for item in combo)
            combinations.append({
                'products': [item['name'] for item in combo],
                'total_cost': total_cost,
                'total_meters': total_meters,
                'cost_per_meter': total_cost / total_meters
            })

    # Find the cheapest combination
    cheapest_option = min(combinations, key=lambda x: x['cost_per_meter'])

    # Find the most convenient option (maximizing meters and minimizing cost)
    convenient_option = min(combinations, key=lambda x: (x['total_cost'], -x['total_meters']))

    # Plot 1: Cheapest Option
    fig, ax = plt.subplots(figsize=(12, 8))
    x_labels = ["-".join(combo['products']) for combo in combinations]
    y_values = [combo['cost_per_meter'] for combo in combinations]

    bars = ax.bar(x_labels, y_values, color='skyblue')

    ax.set_xlabel('Product Combination', fontsize=12)
    ax.set_ylabel('Cost per Meter ($)', fontsize=12)
    ax.set_title('Cheapest Option Comparison (Cost per Meter)', fontsize=14)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)

    for bar, combo in zip(bars, combinations):
        height = bar.get_height()
        if combo == cheapest_option:
            ax.annotate(f'${combo["total_cost"]:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='red')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot1_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    # Plot 2: Most Convenient Option
    fig, ax = plt.subplots(figsize=(12, 8))
    convenient_labels = [f"{combo['products']} (Cost: ${combo['total_cost']:.2f}, Meters: {combo['total_meters']})"
                         for combo in combinations]
    convenient_y_values = [combo['total_meters'] for combo in combinations]

    convenient_bars = ax.bar(convenient_labels, convenient_y_values, color='lightgreen')

    ax.set_xlabel('Product Combination', fontsize=12)
    ax.set_ylabel('Total Meters', fontsize=12)
    ax.set_title('Most Convenient Option (Maximize Meters, Minimize Cost)', fontsize=14)
    ax.set_xticklabels(convenient_labels, rotation=45, ha='right', fontsize=10)

    for bar, combo in zip(convenient_bars, combinations):
        height = bar.get_height()
        if combo == convenient_option:
            ax.annotate(f'${combo["total_cost"]:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='red')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot2_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    # Plot 3: Comparison of Products
    fig, ax = plt.subplots(figsize=(12, 8))

    # Identificar el producto más barato y el más conveniente
    cheapest_product = min(combinations, key=lambda x: x['cost_per_meter'])['products'][0]
    convenient_product = convenient_option['products'][0]

    # Obtener los costos de los productos
    cheapest_product_cost = next(c['total_cost'] for c in combinations if cheapest_product in c['products'])
    convenient_product_cost = next(c['total_cost'] for c in combinations if convenient_product in c['products'])

    # Determinar el producto con el menor costo y el número de múltiplos necesarios
    min_cost_product = cheapest_product if cheapest_product_cost < convenient_product_cost else convenient_product
    max_cost_product = convenient_product if cheapest_product_cost < convenient_product_cost else cheapest_product
    min_cost = min(cheapest_product_cost, convenient_product_cost)
    max_cost = max(cheapest_product_cost, convenient_product_cost)

    # Calcular los múltiplos del producto más barato
    num_multiples = math.ceil(max_cost / min_cost)

    logging.info(f"num_multiples: {num_multiples}")

    # Generar las etiquetas y costos de los múltiplos
    multiples = [f"{min_cost_product}"] + [f"{min_cost_product} x {i}" for i in range(2, num_multiples + 1)]
    multiple_costs = [min_cost * i for i in range(1, num_multiples + 1)]

    # Añadir el producto más caro
    multiples.append(max_cost_product)
    multiple_costs.append(max_cost)

    # Ordenar productos y costos
    sorted_indices = sorted(range(len(multiple_costs)), key=lambda i: multiple_costs[i])
    sorted_multiples = [multiples[i] for i in sorted_indices]
    sorted_costs = [multiple_costs[i] for i in sorted_indices]

    # Crear gráfico de barras
    bars = ax.bar(sorted_multiples, sorted_costs, color='lightcoral')

    # Añadir anotaciones a las barras
    for bar, cost in zip(bars, sorted_costs):
        height = bar.get_height()
        ax.annotate(f'${cost:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Total Cost ($)', fontsize=12)
    ax.set_title('Comparison of Products (Including Multiples)', fontsize=14)
    ax.set_xticklabels(sorted_multiples, rotation=45, ha='right', fontsize=10)

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot3_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    # Respuesta JSON
    return jsonify({
        'cheapest_option': cheapest_option,
        'convenient_option': convenient_option,
        'plot1': plot1_url,
        'plot2': plot2_url,
        'plot3': plot3_url
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
