import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging
from sklearn.datasets import load_breast_cancer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def load_data() -> pd.DataFrame:
    logging.info("Loading breast cancer dataset from sklearn...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def run_preprocessing_pipeline(df: pd.DataFrame):
    logging.info("--- Starting Breast Cancer Preprocessing Pipeline ---")
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Select Top 5 features to simplify the UI
    logging.info("Selecting top 5 features...")
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    logging.info(f"Selected features: {selected_features}")
    
    joblib.dump(selected_features, 'cancer_features.pkl')
    
    X_selected = X[selected_features]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.20, random_state=42, stratify=y
    )
    logging.info(f"Dataset split: 80% Training ({X_train.shape[0]} rows), 20% Testing ({X_test.shape[0]} rows).")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    logging.info("Scaled numerical features using StandardScaler.")
    joblib.dump(scaler, 'cancer_scaler.pkl')
    logging.info("Saved Scaler to 'cancer_scaler.pkl'")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    raw_df = load_data()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(raw_df)
