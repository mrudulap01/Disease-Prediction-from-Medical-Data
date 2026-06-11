import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def load_data(url: str) -> pd.DataFrame:
    try:
        logging.info(f"Loading data from {url}...")
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, header=None, names=columns)
        # Handle '?' values which indicate missing data in this specific dataset
        df.replace('?', np.nan, inplace=True)
        # Binarize the target variable (0 = no disease, 1,2,3,4 = disease)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

def run_preprocessing_pipeline(df: pd.DataFrame):
    logging.info("--- Starting Heart Disease Preprocessing Pipeline ---")
    
    # 1. Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    logging.info(f"Removed duplicates: Shape changed from {initial_shape} to {df.shape}")

    # 2. Impute missing values (ca and thal might have NaNs)
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    logging.info("Handled missing values using median imputation.")
    joblib.dump(imputer, 'heart_imputer.pkl')
    logging.info("Saved Imputer to 'heart_imputer.pkl'")

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )
    logging.info(f"Dataset split: 80% Training ({X_train.shape[0]} rows), 20% Testing ({X_test.shape[0]} rows).")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    logging.info("Scaled numerical features using StandardScaler.")
    joblib.dump(scaler, 'heart_scaler.pkl')
    logging.info("Saved Scaler to 'heart_scaler.pkl'")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    raw_df = load_data(DATASET_URL)
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
