import os
from flask import Flask, jsonify, request
from offline_caption_generator import generate_caption_ar
import requests
from PIL import Image
from io import BytesIO
from flask_cors import CORS

'''
CORS(app, origins=['https://example.com'],
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     headers=['Content-Type', 'Authorization'])
'''

app = Flask(__name__)
CORS(app, resources={r"/predict*": {"origins": "*", 
                                    "methods": ["GET", "POST", "OPTIONS"], 
                                    "headers": ["Content-Type", "Accept"]}})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    image_path = "temp.jpg"
    image.save(image_path)

    caption = generate_caption_ar(image_path) # Load caption from Ai model

    os.remove(image_path)

    return jsonify({'caption': caption})

@app.route('/predict', methods=['GET'])
def predict_url():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'})

    

    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch image'})

    image = Image.open(BytesIO(response.content))
    image_path = "temp.jpg"
    image.save(image_path)

    caption = generate_caption_ar(image_path)  # Load caption from Ai model

    os.remove(image_path)

    return jsonify({'caption': caption})



if __name__ == '__main__':
    app.run(debug=True)
