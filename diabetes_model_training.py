"""
Diabetes Disease Prediction System - Model Training & Evaluation Pipeline

This module trains multiple Machine Learning models, evaluates their performance 
using various metrics, compares them, and selects the best-performing model.

Models Included:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
"""

import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore')

# Import the preprocessing pipeline functions from our previous module
from diabetes_preprocessing import load_data, run_preprocessing_pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple algorithms, evaluates their performance via strict metrics, 
    and compares them to find the top performer.
    
    Returns:
        pd.DataFrame: A dataframe containing the comparison metrics.
        dict: Information regarding the best performing model.
    """
    logging.info("--- Initializing Models ---")
    
    # 1. Define the models to train with stable random states
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Support Vector Machine": SVC(random_state=42, probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "="*70)
    print("Detailed Model Evaluation Metrics")
    print("="*70)

    # 2. Train and Evaluate each model iteratively
    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        
        # Train the model on the training subset
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Predict outcomes on the strictly unseen testing subset
        y_pred = model.predict(X_test)
        
        # 3. Calculate evaluation metrics
        # zero_division=0 prevents warnings if a model fails to predict a certain class
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Collect results for our comparison table
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        
        # 4. Generate Confusion Matrix & Classification Report per model
        print(f"\n[{model_name.upper()}]")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 50)
        
    logging.info("All models trained and evaluated successfully.")
    
    # 5. Build the Comparison DataFrame
    # We sort by F1 Score descending. F1 Score is usually preferred for medical datasets 
    # to balance out False Negatives and False Positives.
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    
    # 6. Automatically select the best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model_obj = trained_models[best_model_name]
    
    best_model_info = {
        "Name": best_model_name,
        "Object": best_model_obj,
        "F1 Score": results_df.iloc[0]["F1 Score"],
        "Accuracy": results_df.iloc[0]["Accuracy"]
    }
    
    return results_df, best_model_info

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training,
    and outputting the comparison metrics.
    """
    DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    
    logging.info("Starting Diabetes Prediction Training Pipeline")
    
    # Step 1: Run the preprocessing pipeline we created previously
    # This automatically detects/replaces zeroes, handles missing values, removes duplicates,
    # caps outliers via IQR, splits the dataset, and applies StandardScaler.
    raw_df = load_data(DATASET_URL)
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
    
    # Step 2: Train and evaluate all 5 requested machine learning models
    results_df, best_model_info = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Step 3: Print the final Model Comparison Table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE (Ranked by F1 Score)")
    print("="*80)
    # Apply clean 4-decimal formatting for professional presentation
    formatters = {
        'Accuracy': lambda x: f"{x:.4f}",
        'Precision': lambda x: f"{x:.4f}",
        'Recall': lambda x: f"{x:.4f}",
        'F1 Score': lambda x: f"{x:.4f}"
    }
    print(results_df.to_string(formatters=formatters))
    
    # Step 4: Announce the selected Best Performing Model
    print("\n" + "*"*80)
    print(f"--- AUTOMATIC SELECTION: BEST PERFORMING MODEL IS '{best_model_info['Name'].upper()}' ---")
    print("*"*80)
    print(f"Reason for Selection: It achieved the highest F1 Score ({best_model_info['F1 Score']:.4f}).")
    print(f"It also achieved an Accuracy of {best_model_info['Accuracy']:.4f}.")
    print("\n[Architectural Note]: For medical diagnosis algorithms, maximizing the F1 score ")
    print("(which is the harmonic mean of Precision and Recall) is generally best practice ")
    print("to strictly minimize highly dangerous False Negatives (undiagnosed diabetics).")

if __name__ == "__main__":
    main()
