# 🫀 A HYBRID MULTI-MODEL BASED APPROACH FOR HEART DISEASE DETECTION AND CLASSIFICATION

## 📌 Overview

This project is our final year capstone project focused on early and accurate heart disease detection using a **multi-modal approach**. We integrated three physiological signals — **ECG (Electrocardiogram)**, **PCG (Phonocardiogram)**, and **PPG (Photoplethysmogram)** — by training separate machine learning models for each signal and then combining their outputs to assess heart disease risk.

## 🧠 Motivation

During our literature review, we observed that most research papers focused on **individual signal types** and mainly aimed at selecting the best-performing model for that signal. This often leads to limited diagnostic accuracy. To address this, we designed a system that leverages the **complementary strengths of all three signals**, offering a more reliable and holistic analysis.

## 🛤️ Project Workflow

1. **Data Collection**: We first collected all the required datasets for model training and validation from **Kaggle datasets** and Google.
2. **Model Training (Jupyter Notebook/Colab)**: We trained the **ECG** model using **Convolutional Neural Network (CNN)** for image processing, **PPG** model using **Random Forest Classifier** for time-series data signals stored as CSV file and for **PCG**, we used **Bidirectional Long Short-Term Memory (Bi-LSTM)** for audio classification.
3. **Model Saving**: After training each model we saved the best-performing model as `best_ecg_cnn.pth` and `ecg_model.h5` for ECG, `scaler.pkl` and `label_encoder.pkl` for PPG and finally `SOUND_LSTM_model.h5`.
4. **Model Integration**: Using the saved model files, we integrated the model and applied a mathematical algorithm to calculate risk score based on confidence score received from each model and converted it to python `.py` file.
5. **Backend Development (Flask API)**: In the above `.py` files we created a **Flask API** that loads all three models and handles inference. It also helps to connect to frontend. When a user uploads data(images, audio and csv) it sends to backend through this API and after computing the output is sent back to frontend as **JSON** file.
6. **Frontend Integration (HTML, CSS, JavaScript)**: Designed a simple frontend layout where users can interact with the model and upload data and receive the output.

## ⚙️ How to Run the Project

Following steps need to be followed:

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

5. **Download virtual environment .venv in this folder**

   ```bash
    python -m venv .venv

6. **Install the requirements file**

   ```bash
   pip install -r requirements.txt

7. **If any error persists for pip**

   ```bash
   pip install --upgrade pip

8. **If any error related to openpyxl**

   ```bash
   pip install openpyxl
   pip install --force-reinstall openpyxl

9. **Activate the virtual environment**

    ```bash
    .venv\Scripts\Activate

10. **Running the backend file may take 2-3 minutes**

    ```bash
    python app.py

11. **In app folder you can start the frontend landing page by clicking on `index.html` and navigate to the Fusion option to access model predictions. (Note: In VS Code, frontend output might not render due to internal preview limitations.)** 

## 🧾 Output and Results 

   ### 🛬 Landing Page Layout

   <p align="center">
  <img src=" ![Image](https://github.com/user-attachments/assets/bbb2f416-0bf0-496b-8665-8f5323fc3fd0)" alt="App Screenshot" width="100%">
</p>
  
   
   ### 🚀 User Interface Video
   
   https://github.com/user-attachments/assets/ed198f7b-32d7-4a57-9298-7fa7144f970c

   ### 🚀 Starting Backend 

   ![Image](https://github.com/user-attachments/assets/c5014288-6344-4c36-9472-43d8d5777917)

   ### 🌐 Frontend Input

   ![Image](https://github.com/user-attachments/assets/88888667-dc1c-4ab0-90ab-febbff4c48c8)

   ### After clicking "Predict":
   ### 📤 Backend Output

   ![Image](https://github.com/user-attachments/assets/153c1f93-d2b1-44c2-8a4e-9978d76578c8)

   ### 🔎 Frontend Output

   ![Image](https://github.com/user-attachments/assets/a60c5484-a2f6-402a-9bc6-e65c2929ffc2)
  
   ### 📊 Each Model Accuracy 
   
   | Signal Type | Model Used                          | Accuracy                                  |
   |-------------|-------------------------------------|-------------------------------------------|
   | ECG         | CNN                                 | 93.32%                                    |
   | PPG         | Random Forest                       | 96%                                       |
   | PCG         | BiLSTM                              | 96%                                       |

   

<p align="center">!!! THANK YOU !!!</p>


📩 For any queries, feel free to reach out at:
aryanmatte2023@gmail.com


