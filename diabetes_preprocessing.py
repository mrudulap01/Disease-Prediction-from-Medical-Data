"""
Diabetes Disease Prediction System - Data Preprocessing Pipeline

This module performs the complete data preprocessing pipeline to prepare 
the PIMA Indians Diabetes dataset for Machine Learning modeling. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
import sys

# Configure basic logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_data(url: str) -> pd.DataFrame:
    """Loads dataset from URL."""
    try:
        logging.info(f"Loading data from {url}...")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

def run_preprocessing_pipeline(df: pd.DataFrame):
    """
    Executes the complete preprocessing pipeline on the given dataset.
    
    Steps included:
    1. Detect invalid values.
    2. Handle missing values correctly.
    3. Remove duplicates.
    4. Detect outliers using IQR.
    5. Split dataset into Training/Testing.
    6. Scale numerical features using StandardScaler.
    """
    logging.info("--- Starting Preprocessing Pipeline ---")
    
    # ---------------------------------------------------------
    # Step 1: Remove duplicates
    # ---------------------------------------------------------
    initial_shape = df.shape
    df = df.drop_duplicates()
    logging.info(f"Removed duplicates: Shape changed from {initial_shape} to {df.shape}")

    # ---------------------------------------------------------
    # Step 2: Detect invalid values (zeros in biological features)
    # ---------------------------------------------------------
    # In the PIMA dataset, a value of 0 in the following columns is biologically 
    # impossible and actually indicates a missing value.
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in invalid_zero_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            logging.info(f"Detected {zero_count} invalid zero values in '{col}'. Replacing with NaN.")
            # Replacing 0 with NaN so our imputer can recognize them as missing data
            df[col] = df[col].replace(0, np.nan)

    # ---------------------------------------------------------
    # Step 3: Handle missing values correctly
    # ---------------------------------------------------------
    # We use SimpleImputer with a 'median' strategy because median is robust 
    # against outliers, ensuring extreme values don't heavily skew our imputations.
    imputer = SimpleImputer(strategy='median')
    
    # Fit and transform the columns with missing values
    df[invalid_zero_cols] = imputer.fit_transform(df[invalid_zero_cols])
    logging.info("Handled missing values using median imputation.")
    
    # Save the imputer for deployment
    joblib.dump(imputer, 'diabetes_imputer.pkl')
    logging.info("Saved Imputer to 'diabetes_imputer.pkl'")

    # ---------------------------------------------------------
    # Step 4: Detect outliers using IQR (Interquartile Range)
    # ---------------------------------------------------------
    # We detect outliers using the IQR method. To prevent data loss, we cap 
    # the extreme values at the upper and lower bounds instead of deleting the rows.
    numerical_features = df.columns.drop('Outcome')
    
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for typical data
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count potential outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            logging.info(f"Detected {outliers} outliers in '{col}' using IQR. Capping them.")
            
            # Capping the outliers to the calculated bounds
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # ---------------------------------------------------------
    # Step 5: Split dataset into 80% Training and 20% Testing
    # ---------------------------------------------------------
    # Separate independent features (X) and dependent target variable (y)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Split the data using random_state for reproducibility
    # Using stratify=y is best practice to keep the same class proportions in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20,      # 20% for testing
        random_state=42,     # Ensures reproducible splits every time
        stratify=y           # Recommended for binary classification
    )
    logging.info(f"Dataset split: 80% Training ({X_train.shape[0]} rows), 20% Testing ({X_test.shape[0]} rows).")

    # ---------------------------------------------------------
    # Step 6: Scale numerical features using StandardScaler
    # ---------------------------------------------------------
    # Scaling is performed *after* splitting to prevent data leakage (information
    # from the test set leaking into the training step).
    scaler = StandardScaler()
    
    # Fit the scaler on training data ONLY, then transform both train and test sets
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    logging.info("Scaled numerical features using StandardScaler.")
    
    # Save the scaler for deployment
    joblib.dump(scaler, 'diabetes_scaler.pkl')
    logging.info("Saved Scaler to 'diabetes_scaler.pkl'")
    
    logging.info("--- Preprocessing Pipeline Completed Successfully ---")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    
    # 1. Load raw data
    raw_df = load_data(DATASET_URL)
    
    # 2. Execute full preprocessing pipeline
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
    
    # 3. Quick verification and preview
    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    print("Training Features (First 2 rows, scaled):\n", X_train.head(2))
    print("\nTraining Target Class Distribution:\n", y_train.value_counts())
    print("============================================================")
