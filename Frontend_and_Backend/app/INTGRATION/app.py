import os
import numpy as np
import librosa
import torch
import cv2
import pickle
import joblib
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)
CORS(app)
# app = Flask(__name__)  # Define the app first
# CORS(app, resources={r"/predict": {"origins": "http://172.20.192.179:5500"}})  # Then apply CORS


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Load PyTorch ECG Model**
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(128, 64)  
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_path = "best_ecg_cnn.pth"
num_classes = 4
ecg_model = ECG_CNN(num_classes).to(device)

if os.path.exists(model_path):
    try:
        ecg_model.load_state_dict(torch.load(model_path, map_location=device))
        ecg_model.eval()
        print(f"✅ ECG Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading ECG model: {e}")
else:
    raise FileNotFoundError(f"❌ Trained ECG model file '{model_path}' not found!")

# **Transform for ECG Image**
ecg_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# **Class Labels**
class_labels = {0: "MI", 1: "PMI", 2: "Abnormal Heart Beats", 3: "Normal"}

# **Load Other Models**
MODELS = {}
try:
    MODELS["audio"] = load_model("SOUND_LSTM_model.h5", compile=False)
    print("✅ Audio model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Audio model: {e}")

try:
    MODELS["ppg"] = joblib.load("best_ppg_model.pkl")
    print("✅ PPG model loaded successfully")
except Exception as e:
    print(f"❌ Error loading PPG model: {e}")

try:
    MODELS["scaler"] = joblib.load("scaler.pkl")
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading Scaler: {e}")

# **Risk Weights**
RISK_WEIGHTS = {
    "ECG": {"MI": 0.9, "PMI": 0.7, "Abnormal Heart Beats": 0.6, "Normal": 0.1},
    "AUDIO": {"Healthy": 0.1, "Unhealthy": 0.9},
    "PPG": {0: 0.1, 1: 0.9}
}

# **Preprocessing Functions**
def preprocess_audio(file_path, max_pad_len=400):
    audio, sample_rate = librosa.load(file_path, sr=2000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max(0, pad_width))), mode='constant')
    return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

def preprocess_ppg(data_row):
    # Convert input to DataFrame if needed
    if isinstance(data_row, pd.Series):
        data_row = data_row.to_frame().T  # Convert Series to DataFrame (1 row)

    # Drop label column if it exists
    if "Label" in data_row:
        data_row = data_row.drop(columns=["Label"])

    # Ensure the input is correctly reshaped
    processed_ppg = MODELS["scaler"].transform(data_row.values)  # Extract values and scale

    return processed_ppg  # Return properly shaped 2D array


# **Prediction Functions**
def predict_audio(file_path):
    features = preprocess_audio(file_path)
    prediction = MODELS["audio"].predict(features)[0]
    return ("Unhealthy", prediction) if prediction > 0.5 else ("Healthy", 1 - prediction)

def predict_ecg(file_path):
    image = Image.open(file_path).convert("L")
    img_tensor = ecg_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = ecg_model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    
    return class_labels.get(predicted_class.item(), "Unknown"), float(torch.max(F.softmax(output, dim=1)).item())

def predict_ppg(ppg_row):
    processed_ppg = preprocess_ppg(ppg_row)
    prediction_prob = MODELS["ppg"].predict_proba(processed_ppg)[:, 1][0]
    return (0, prediction_prob) if prediction_prob > 0.5 else (1, prediction_prob)


def calculate_risk(audio_result, audio_confidence, ecg_result, ecg_confidence, ppg_result, ppg_confidence):
    risk_score = (
        RISK_WEIGHTS["AUDIO"][audio_result] * audio_confidence +
        RISK_WEIGHTS["ECG"].get(ecg_result, 0) * ecg_confidence +
        RISK_WEIGHTS["PPG"][ppg_result] * ppg_confidence
    ) / 3

    return round(float(risk_score), 2)  # Convert NumPy array to Python float


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded files
        ecg_file = request.files.get("ecg_image")
        ppg_file = request.files.get("ppg_file")
        audio_file = request.files.get("audio_file")

        if not (ecg_file and ppg_file and audio_file):
            return jsonify({"error": "Missing files"}), 400

        # Save files temporarily
        ecg_path = os.path.join(UPLOAD_FOLDER, secure_filename(ecg_file.filename))
        ppg_path = os.path.join(UPLOAD_FOLDER, secure_filename(ppg_file.filename))
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))

        ecg_file.save(ecg_path)
        ppg_file.save(ppg_path)
        audio_file.save(audio_path)

        # Make Predictions
        audio_result, audio_confidence = predict_audio(audio_path)
        ecg_result, ecg_confidence = predict_ecg(ecg_path)

        # Load PPG Data (assuming CSV)
        ppg_df = pd.read_csv(ppg_path)  
        ppg_result, ppg_confidence = predict_ppg(ppg_df.iloc[0])  # Use first row of PPG file

        # Calculate Final Risk Score
        final_risk_score = calculate_risk(audio_result, audio_confidence, ecg_result, ecg_confidence, ppg_result, ppg_confidence)

        # Return Response
        result = {
            "Audio Result": audio_result,
            "Audio Confidence": round(float(audio_confidence), 2),  
            "ECG Result": ecg_result,
            "ECG Confidence": round(float(ecg_confidence), 2),
            "PPG Result": "MI" if ppg_result == 1 else "Normal",
            "PPG Confidence": round(float(ppg_confidence), 2),
            "Final Risk Score": round(float(final_risk_score), 2)
        }

        print("Returning result:", result)  # Debugging print
        return jsonify(result)

    except Exception as e:
        print("Error:", str(e))  # Print error in the console
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


