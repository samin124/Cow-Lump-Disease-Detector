import torch
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_image_bytes

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = load_model(MODEL_PATH, DEVICE)
print("Model loaded!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # You need to create this template

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        label, confidence = predict_image_bytes(model, image_bytes, DEVICE)
        os.remove(filepath)
        return jsonify({'label': label, 'confidence': round(confidence, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or fallback to 5000
    app.run(host='0.0.0.0', port=port)

