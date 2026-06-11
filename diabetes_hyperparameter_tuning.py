"""
Diabetes Disease Prediction System - Hyperparameter Tuning

This module focuses on improving the performance of the best Machine Learning model
(Random Forest) through extensive hyperparameter tuning using GridSearchCV.

It compares baseline performance against the tuned model and saves the final 
production-ready model to disk.
"""

import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

# Suppress warnings for cleaner production output
warnings.filterwarnings('ignore')

# Import our preprocessing logic
from diabetes_preprocessing import load_data, run_preprocessing_pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def tune_model(X_train, y_train, X_test, y_test):
    """
    Executes GridSearchCV to find the optimal hyperparameters for Random Forest.
    Compares baseline performance to the tuned performance.
    
    Returns:
        RandomForestClassifier: The optimal, tuned model.
    """
    logging.info("--- Phase 1: Baseline Model Evaluation ---")
    
    # 1. Train the baseline XGBoost (Default Parameters)
    baseline_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    baseline_xgb.fit(X_train, y_train)
    y_pred_base = baseline_xgb.predict(X_test)
    
    base_accuracy = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, zero_division=0)
    
    logging.info(f"Baseline Accuracy: {base_accuracy:.4f} | Baseline F1: {base_f1:.4f}")
    
    logging.info("--- Phase 2: Hyperparameter Tuning via GridSearchCV ---")
    
    # 2. Define the hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    # 3. Initialize GridSearchCV with 5-fold cross-validation
    # We optimize for 'f1' due to the medical nature of the dataset (balance of precision/recall)
    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid=param_grid,
        cv=5,                 # 5-fold cross validation
        scoring='f1',         # Target metric
        n_jobs=-1,            # Use all available CPU cores for speed
        verbose=1             # Print progress updates
    )
    
    logging.info("Starting 5-Fold Grid Search... this may take a moment.")
    xgb_grid.fit(X_train, y_train)
    
    # 4. Extract the best model and parameters
    best_xgb = xgb_grid.best_estimator_
    best_params = xgb_grid.best_params_
    
    logging.info("Grid Search Completed.")
    
    # 5. Evaluate the Tuned Model on the unseen testing dataset
    y_pred_tuned = best_xgb.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
    
    # 6. Display Comparison and Results
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS: XGBOOST")
    print("="*80)
    
    print("\n[Best Hyperparameters Found]")
    for param, value in best_params.items():
        print(f" - {param}: {value}")
        
    print("\n[Performance Comparison (Testing Data)]")
    
    # Creating a comparison dataframe for clean display
    comparison_data = {
        "Metric": ["Accuracy", "F1 Score"],
        "Baseline Model": [f"{base_accuracy:.4f}", f"{base_f1:.4f}"],
        "Tuned Model": [f"{tuned_accuracy:.4f}", f"{tuned_f1:.4f}"],
        "Improvement": [
            f"{(tuned_accuracy - base_accuracy):+.4f}", 
            f"{(tuned_f1 - base_f1):+.4f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    print("\n[Tuned Model Classification Report]")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))
    print("="*80)
    
    return best_xgb

def save_model(model, filepath="best_diabetes_model.pkl"):
    """Saves the trained Machine Learning model to disk."""
    try:
        joblib.dump(model, filepath)
        logging.info(f"--- Model Successfully Saved to '{filepath}' ---")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")

def main():
    """Main execution function for the tuning pipeline."""
    DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    logging.info("Starting Diabetes Prediction Model Optimization Pipeline")
    
    # 1. Load and Preprocess Data
    raw_df = load_data(DATASET_URL)
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
    
    # 2. Tune XGBoost Model
    tuned_model = tune_model(X_train, y_train, X_test, y_test)
    
    # 3. Save the tuned model for future inference
    save_model(tuned_model)

if __name__ == "__main__":
    main()
