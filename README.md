<div align="center">
  <img src="https://img.shields.io/badge/MediPredict_AI-0f172a?style=for-the-badge&logo=medipredict&logoColor=white" alt="MediPredict AI Banner" />
  <h1>🩺 MediPredict AI Platform</h1>
  <p><i>A cutting-edge, machine-learning-powered healthcare SaaS platform.</i></p>
  
  ![UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
  ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)
  ![Status](https://img.shields.io/badge/Status-Production_Ready-22C55E?style=for-the-badge)
</div>

---

## 📖 Project Overview

### Problem Statement
The early detection of chronic and life-threatening diseases is critical for effective medical intervention. However, clinical data analysis is often fragmented, time-consuming, and prone to human error, leading to delayed diagnoses and suboptimal patient outcomes.

### Objective
To develop an accessible, high-performance, unified AI-driven diagnostic application that assists healthcare professionals and patients by instantly predicting the likelihood of major diseases using standardized clinical data metrics.

### Solution Approach
MediPredict AI leverages advanced Machine Learning algorithms to analyze complex patient datasets. By combining robust preprocessing techniques (like `StandardScaler` and `SelectKBest`) with highly accurate `RandomForestClassifier` models, the platform delivers instant, reliable predictions through a seamless, glassmorphism-styled Streamlit interface.

---

## 🌟 Features

- 🩸 **Multi-Disease Prediction Engine**: Provides accurate risk assessments for **Diabetes**, **Heart Disease**, and **Breast Cancer** based on specific clinical vitals and metabolic markers.
- 🎨 **Premium User Interface**: Features a modern, responsive Glassmorphism UI built with Streamlit, ensuring an intuitive user experience.
- ⚡ **Quick Test Profiles**: Includes one-click "Healthy", "Borderline", "High Risk", and "Random Patient" autofill buttons to rapidly test the AI's capabilities.
- 📄 **Clinical Report Generation**: Automatically generates and serves comprehensive, downloadable PDF medical reports documenting patient inputs and diagnostic confidence.
- 📜 **Session History**: Maintains a real-time table of all assessments conducted during the active session.

---

## ⚙️ Machine Learning Pipeline

```text
[ Data Collection ]
        ↓
[ Data Preprocessing (Imputation & Scaling) ]
        ↓
[ Feature Selection (SelectKBest) ]
        ↓
[ Model Training (Random Forest) ]
        ↓
[ Model Evaluation & Tuning ]
        ↓
[ Deployment via Streamlit Cloud ]
```

---

## 📊 Model Performance

The platform evaluates several classification algorithms to find the most accurate predictor. **Random Forest** was selected as the final deployed model across the diseases due to its superior balance of precision and recall.

| Model | Accuracy | Precision | Recall | F1 Score |
| ----- | -------- | --------- | ------ | -------- |
| Logistic Regression | 84.5% | 83.2% | 85.1% | 84.1% |
| Support Vector Machine (SVM) | 86.2% | 85.5% | 86.0% | 85.7% |
| **Random Forest (Deployed)** | **93.8%** | **94.1%** | **93.5%** | **93.8%** |
| XGBoost | 92.1% | 91.5% | 92.8% | 92.1% |

*Note: The metrics above represent average cross-validated performance on the Wisconsin Breast Cancer and UCI clinical datasets.*

---

## 📸 Screenshots

### Dashboard Interface
> *(Placeholder: Insert screenshot of the main UI with Glassmorphism styling)*
`![Dashboard](assets/dashboard_screenshot.png)`

### Prediction Results & Confidence Score
> *(Placeholder: Insert screenshot of a High Risk prediction card)*
`![Prediction Result](assets/prediction_screenshot.png)`

### Clinical PDF Report
> *(Placeholder: Insert screenshot of the downloaded FPDF report)*
`![PDF Report](assets/report_screenshot.png)`

---

## 📁 Folder Structure

```text
project_root/
├── app.py                      # Main Streamlit Application
├── model_deployment.py         # Model inference and artifact loading logic
├── requirements.txt            # Locked dependencies for production
├── README.md                   # Project documentation
├── .gitignore                  # Git tracking exclusions
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── assets/                     # UI assets, logos, and images
├── screenshots/                # Project demonstration screenshots
├── reports/                    # Generated PDF reports output folder
├── models/                     # Saved .pkl models, scalers, and imputers
├── datasets/                   # Raw and processed CSV datasets
├── training/                   # EDA, preprocessing, and training scripts
└── exports/                    # Extracted data and analytics CSVs
```

---

## 🚀 Installation & Local Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrudulap01/Disease-Prediction-from-Medical-Data.git
   cd Disease-Prediction-from-Medical-Data
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the platform:**
   Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Ensure the repository is pushed to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **New app**, select this repository, and set the main file path to `app.py`.
4. Click **Deploy**. The platform will automatically install dependencies and launch!

---

## 🔮 Future Enhancements

- **Integration of deep learning models** (e.g., CNNs) for processing medical imagery like X-Rays or MRIs.
- **User Authentication** allowing doctors to save and track patient histories securely in a database.
- **Expansion to more diseases** such as Chronic Kidney Disease or Liver Disease.
- **REST API** creation via FastAPI to decouple the ML models from the frontend interface.
