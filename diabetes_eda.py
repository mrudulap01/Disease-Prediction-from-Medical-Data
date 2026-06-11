"""
Diabetes Disease Prediction System - Exploratory Data Analysis (EDA)

This module performs complete exploratory data analysis on the PIMA Indians Diabetes Dataset.
It generates statistical summaries and visualization plots to uncover insights from the data.

Sections:
    1. Data Loading
    2. Data Exploration
    3. Data Visualization
    4. Data Preprocessing (Scikit-Learn)
    5. Insights
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configure basic logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Constants
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
OUTPUT_DIR = "eda_outputs"


def load_data(url: str) -> pd.DataFrame:
    """
    Section 1: Data Loading
    Loads the PIMA Indians Diabetes dataset from a given URL or file path.
    
    Args:
        url (str): The URL or local path to the dataset CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logging.info("--- Section 1: Data Loading ---")
    try:
        logging.info(f"Loading data from {url}...")
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load the dataset: {e}")
        sys.exit(1)


def explore_data(df: pd.DataFrame) -> None:
    """
    Section 2: Data Exploration
    Explores the dataset to display shape, data types, missing values, and statistical summary.
    
    Args:
        df (pd.DataFrame): The dataset to explore.
    """
    logging.info("--- Section 2: Data Exploration ---")
    
    # 2.1 Dataset Shape
    print("\n" + "="*40)
    print("1. Dataset Shape:")
    print("="*40)
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    
    # 2.2 Data Types
    print("\n" + "="*40)
    print("2. Data Types:")
    print("="*40)
    print(df.dtypes)
    
    # 2.3 Missing Values
    print("\n" + "="*40)
    print("3. Missing Values:")
    print("="*40)
    # Check for actual nulls
    print("Null values per column:")
    print(df.isnull().sum())
    
    # In PIMA dataset, zeroes in certain biological parameters represent missing data
    columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print(f"\nNote: In the PIMA dataset, '0' values in certain columns often indicate missing data.")
    for col in columns_with_zero_as_missing:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f" - {col} has {zero_count} zero values (potentially missing).")
            
    # 2.4 Statistical Summary
    print("\n" + "="*40)
    print("4. Statistical Summary:")
    print("="*40)
    print(df.describe().T)
    print("\n")


def visualize_data(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """
    Section 3: Data Visualization
    Generates various plots (histograms, boxplots, correlation matrix, class distribution)
    and saves them to the specified output directory.
    
    Args:
        df (pd.DataFrame): The dataset to visualize.
        output_dir (str): Directory to save the generated plots.
    """
    logging.info("--- Section 3: Data Visualization ---")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Visualizations will be saved to the '{output_dir}' directory.")
    
    # Set plotting style
    sns.set_theme(style="whitegrid")
    
    try:
        # 3.1 Class Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Outcome', palette='viridis', hue='Outcome', legend=False)
        plt.title("Class Distribution (0: Non-Diabetic, 1: Diabetic)")
        plt.xlabel("Outcome (Diabetes)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        class_dist_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(class_dist_path)
        plt.close()
        logging.info(f"Saved: {class_dist_path}")

        # 3.2 Histograms
        df.hist(bins=20, figsize=(15, 12), color='teal', edgecolor='black')
        plt.suptitle("Histograms of All Features", y=1.02, fontsize=16)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, 'histograms.png')
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Saved: {hist_path}")

        # 3.3 Boxplots (Outlier detection)
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=df, orient="h", palette="Set2")
        plt.title("Boxplots of All Features (Outlier Detection)", fontsize=16)
        plt.tight_layout()
        boxplot_path = os.path.join(output_dir, 'boxplots.png')
        plt.savefig(boxplot_path)
        plt.close()
        logging.info(f"Saved: {boxplot_path}")

        # 3.4 Correlation Matrix
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix", fontsize=16)
        plt.tight_layout()
        corr_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(corr_path)
        plt.close()
        logging.info(f"Saved: {corr_path}")
        
    except Exception as e:
        logging.error(f"Error during visualization generation: {e}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 4: Data Preprocessing (Scikit-Learn)
    Demonstrates the use of Scikit-Learn for handling missing values and feature scaling.
    
    Args:
        df (pd.DataFrame): The dataset to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    logging.info("--- Section 4: Data Preprocessing (Scikit-Learn) ---")
    df_processed = df.copy()
    
    # 4.1 Replace 0 with NaN for relevant columns
    columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed[columns_with_zero_as_missing] = df_processed[columns_with_zero_as_missing].replace(0, np.nan)
    logging.info(f"Replaced 0 with NaN in columns: {columns_with_zero_as_missing}")
    
    # 4.2 Impute missing values using Scikit-Learn
    # We use median for imputation as it's robust to outliers
    imputer = SimpleImputer(strategy='median')
    df_processed[columns_with_zero_as_missing] = imputer.fit_transform(df_processed[columns_with_zero_as_missing])
    logging.info("Imputed missing values using SimpleImputer (median strategy).")
    
    # 4.3 Feature Scaling using Scikit-Learn
    # We scale all features except the target 'Outcome'
    features = df_processed.drop('Outcome', axis=1).columns
    scaler = StandardScaler()
    df_processed[features] = scaler.fit_transform(df_processed[features])
    logging.info("Scaled features using StandardScaler.")
    
    print("\n" + "="*40)
    print("Preprocessed Data (First 5 Rows):")
    print("="*40)
    print(df_processed.head())
    print("\n")
    
    return df_processed


def generate_insights() -> None:
    """
    Section 5: Insights
    Displays key actionable insights derived from the EDA.
    """
    logging.info("--- Section 5: Key Insights ---")
    print("\n" + "="*40)
    print("Actionable Insights from EDA:")
    print("="*40)
    
    insights = [
        "1. Missing Data (Hidden as Zeros): Features like Insulin, SkinThickness, BloodPressure, BMI, and Glucose have invalid '0' values that need imputation (e.g., median/mean replacement) before modeling.",
        "2. Class Imbalance: The target 'Outcome' is imbalanced. There are significantly more negative cases (0) than positive cases (1). Consider techniques like SMOTE or class weighting during model training.",
        "3. Outliers: Boxplots reveal significant outliers in 'Insulin' and 'DiabetesPedigreeFunction'. Robust scaling or outlier treatment may be necessary for distance-based ML models.",
        "4. Feature Correlation: 'Glucose', 'BMI', and 'Age' show the highest positive correlation with the target variable 'Outcome', making them strong predictors.",
        "5. Skewness: Several features like 'Insulin', 'Age', and 'DiabetesPedigreeFunction' are right-skewed. Log or Box-Cox transformations might improve model performance."
    ]
    
    for insight in insights:
        print(insight)
    print("\n")


def main():
    """
    Main execution function to run the complete EDA pipeline.
    """
    logging.info("Starting Diabetes Disease Prediction System - EDA Pipeline")
    
    # 1. Load Data
    df = load_data(DATASET_URL)
    
    # 2. Explore Data
    explore_data(df)
    
    # 3. Visualize Data
    visualize_data(df)
    
    # 4. Preprocess Data
    df_preprocessed = preprocess_data(df)
    
    # 5. Generate Insights
    generate_insights()
    
    logging.info("EDA Pipeline completed successfully.")


if __name__ == "__main__":
    main()
