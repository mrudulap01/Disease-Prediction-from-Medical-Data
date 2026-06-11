import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from heart_disease_preprocessing import load_data, run_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_model():
    # Load and preprocess
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    raw_df = load_data(DATASET_URL)
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
    
    logging.info("Training Heart Disease Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {acc:.4f}")
    logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save the model
    joblib.dump(model, 'best_heart_model.pkl')
    logging.info("Saved trained model to 'best_heart_model.pkl'")

if __name__ == "__main__":
    train_model()
