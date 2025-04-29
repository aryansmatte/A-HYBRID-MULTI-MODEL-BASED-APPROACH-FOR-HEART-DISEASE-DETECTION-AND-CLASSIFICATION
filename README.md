# 🫀 A HYBRID MULTI-MODEL BASED APPROACH FOR HEART DISEASE DETECTION AND CLASSIFICATION

## 📌 Overview

This project is our final year capstone project focused on early and accurate heart disease detection using a **multi-modal approach**. We integrated three physiological signals — **ECG (Electrocardiogram)**, **PCG (Phonocardiogram)**, and **PPG (Photoplethysmogram)** — by training separate machine learning models for each signal and then combining their outputs to assess heart disease risk.

## 🧠 Motivation

During our literature review, we observed that most research papers focused on **individual signal types** and mainly aimed at selecting the best-performing model for that signal. This often leads to limited diagnostic accuracy. To address this, we designed a system that leverages the **complementary strengths of all three signals**, offering a more reliable and holistic analysis.

## 🛤️ Project Workflow

1. **Data Collection**: Firstly we collected all the required dataset for model training and validtion from **Kaggle datasets** present on google.
2. **Model Training (Jupyter Notebook/Colab)**: We trained the **ECG** model using **Convolutional Neural Network (CNN)** for image processing, **PPG** model using **Random Forest Classifier** for time-series data signals stored as csv file and **PCG** we used **Bidirectional Long Short-Term Memory (Bi-LSTM)** for audio classification.
3. **Model Saving**: After training each model we saved the best observed model as `best_ecg_cnn.pth` and `ecg_model.h5` for ECG, `scaler.pkl` and `label_encoder.pkl` for PPG and finally `SOUND_LSTM_model.h5`.
4. **Model Integration**: Then by using the above model files we integrated the model and add mathematical algorithm to calculate risk score based on confidence score received from each model and converted it to python `.py` file.
5. **Backend Development (Flask API)**: In the above `.py` files we created a **Flask API** that loads all three models and handles inference. It also helps to connect to frontend and when user uploads data(images, audio and csv) it sends to backend through this API and after computing the output is send back to frontend as **JSON** file.
6. **Frontend Integration (HTML, CSS, JavaScript)**: Designed a simple frontend layout where users can interact with the model and upload data and get output.

## ⚙️ How to Run the Project

Follow the following steps if you want to try the model on your own:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/aryansmatte/A-HYBRID-MULTI-MODEL-BASED-APPROACH-FOR-HEART-DISEASE-DETECTION-AND-CLASSIFICATION

2. **Navigate to Frontend_and_Backend folder**

   ```bash
   cd Frontend_and_Backend

3. **Move to app folder which consist of frontend landing page index.html**

   ```bash
   cd app
   
4. **Navigate to INTGRATION folder which has main model code**

   ```bash
   cd INTGRATION

5. **Download virtual environment .venv int this folder**

   ```bash
    python -m venv .venv

6. **Install the requiremnts file**

   ```bash
   pip install -r requirements.txt

7. **If any error persist for pip**

   ```bash
   pip install --upgrade pip

8. **If any error related to openpyxl**

   ```bash
   pip install openpyxl
   pip install --force-reinstall openpyxl

9. **Activate the virtual environment**

    ```bash
    .venv\Scripts\Activate

10. **Run the backend file may take 2-3 minutes**

    ```bash
    python app.py

11. **In app folder you cn start the frontend landing page by clicking on it and navigate to Fusion option there to go to the model implementation and VScode might give some error so directly open the index.html from folder with chrome** 


