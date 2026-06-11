"""
Unified Disease Prediction System - Deployment Module
Provides production-ready functions to load models and perform inference for Diabetes, Heart Disease, and Breast Cancer.
"""

import joblib
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- DIABETES ---
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
DIABETES_ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def load_diabetes_model():
    try:
        return joblib.load("best_diabetes_model.pkl"), joblib.load("diabetes_imputer.pkl"), joblib.load("diabetes_scaler.pkl")
    except Exception as e:
        logging.error(f"Failed to load diabetes artifacts: {e}")
        return None, None, None

def predict_diabetes(user_input: dict, artifacts: tuple) -> dict:
    model, imputer, scaler = artifacts
    df_input = pd.DataFrame([user_input])[DIABETES_FEATURES]
    for col in DIABETES_ZERO_COLS:
        if df_input[col].iloc[0] == 0:
            df_input[col] = np.nan
    df_input[DIABETES_ZERO_COLS] = imputer.transform(df_input[DIABETES_ZERO_COLS])
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=DIABETES_FEATURES)
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    return {
        "prediction_code": int(prediction),
        "prediction_label": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "confidence_score": f"{confidence:.2f}%",
        "probabilities": {"Non-Diabetic": f"{probabilities[0]*100:.2f}%", "Diabetic": f"{probabilities[1]*100:.2f}%"}
    }


# --- HEART DISEASE ---
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def load_heart_model():
    try:
        return joblib.load("best_heart_model.pkl"), joblib.load("heart_imputer.pkl"), joblib.load("heart_scaler.pkl")
    except Exception as e:
        logging.error(f"Failed to load heart artifacts: {e}")
        return None, None, None

def predict_heart_disease(user_input: dict, artifacts: tuple) -> dict:
    model, imputer, scaler = artifacts
    df_input = pd.DataFrame([user_input])[HEART_FEATURES]
    
    # Imputation for any potential NaNs (ca, thal)
    df_input = pd.DataFrame(imputer.transform(df_input), columns=HEART_FEATURES)
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=HEART_FEATURES)
    
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    return {
        "prediction_code": int(prediction),
        "prediction_label": "Heart Disease Present" if prediction == 1 else "No Heart Disease",
        "confidence_score": f"{confidence:.2f}%",
        "probabilities": {"No Heart Disease": f"{probabilities[0]*100:.2f}%", "Heart Disease Present": f"{probabilities[1]*100:.2f}%"}
    }


# --- BREAST CANCER ---
# Features were dynamically selected during training. We need to load them.
def load_cancer_model():
    try:
        features = joblib.load("cancer_features.pkl")
        return joblib.load("best_cancer_model.pkl"), features, joblib.load("cancer_scaler.pkl")
    except Exception as e:
        logging.error(f"Failed to load breast cancer artifacts: {e}")
        return None, None, None

def predict_breast_cancer(user_input: dict, artifacts: tuple) -> dict:
    model, features, scaler = artifacts
    df_input = pd.DataFrame([user_input])[features]
    
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=features)
    
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    return {
        "prediction_code": int(prediction),
        "prediction_label": "Malignant (Cancer)" if prediction == 0 else "Benign (No Cancer)",
        "confidence_score": f"{confidence:.2f}%",
        "probabilities": {"Malignant": f"{probabilities[0]*100:.2f}%", "Benign": f"{probabilities[1]*100:.2f}%"}
    }
