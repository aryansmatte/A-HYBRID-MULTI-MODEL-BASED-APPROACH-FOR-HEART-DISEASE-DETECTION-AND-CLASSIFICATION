import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import sys

# Ensure the model file is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model
try:
    from model import ECG_CNN  # Ensure model.py exists
except ModuleNotFoundError:
    raise ImportError("Error: 'model.py' not found. Ensure the model file is in the correct directory.")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# **Device Setup**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Load Trained Model**
model_path = "best_ecg_cnn.pth"
num_classes = 4  # Adjust based on dataset
model = ECG_CNN(num_classes).to(device)

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
else:
    raise FileNotFoundError(f"Trained model file '{model_path}' not found!")

# **Define Image Preprocessing**
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# **Class Labels**
class_labels = {0: "Myocardial", 1: "MI", 2: "Abnormal Heart Beat", 3: "Normal"}

# **Prediction Route**
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid file format. Only PNG, JPG, and JPEG are allowed."}), 400

    try:
        # Load image and preprocess
        image = Image.open(file).convert("L")  # Convert to grayscale
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)

        result = class_labels.get(predicted_class.item(), "Unknown")

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# **Run the Flask App**
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500, debug=True)  # Make accessible over LAN
