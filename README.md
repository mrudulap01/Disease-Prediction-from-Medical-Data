# 🩺 MediPredict AI Platform

![UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)

**MediPredict AI** is a cutting-edge, machine-learning-powered healthcare SaaS platform. It allows users to predict the likelihood of three major diseases based on clinical data using advanced Random Forest Classifiers. The platform features a premium, glassmorphism-styled UI and automatically generates downloadable PDF clinical reports.

## 🚀 Features

- **Multi-Disease Prediction Engine**:
  - 🩸 **Diabetes**: Predicts risk based on demographics and metabolic markers (Glucose, Insulin, BMI, etc.).
  - ❤️ **Heart Disease**: Predicts risk using clinical vitals and test results (Cholesterol, ECG, Max Heart Rate, etc.).
  - 🔬 **Breast Cancer**: Predicts malignancy using precise cell nucleus geometry (Radius, Perimeter, Concave Points).
- **Premium User Interface**: Modern glassmorphism UI built with Streamlit, featuring an intuitive grid layout, soft animations, and dynamic repainting based on the selected disease.
- **Quick Test Profiles**: Instantly load "Healthy", "Borderline", "High Risk", or completely "Random" patient profiles to test the AI without dragging sliders manually.
- **Clinical Report Generation**: Generates comprehensive, downloadable PDF reports documenting the patient's inputs and the AI's diagnostic confidence score.
- **Prediction History**: Tracks all assessments made during the current session.

## 🛠️ Technology Stack

- **Frontend**: Streamlit (with custom CSS injections for Glassmorphism styling)
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `SelectKBest`, `StandardScaler`)
- **Data Manipulation**: Pandas, NumPy
- **Report Generation**: FPDF2

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrudulap01/Disease-Prediction-from-Medical-Data.git
   cd Disease-Prediction-from-Medical-Data
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the platform:**
   Open your browser and navigate to `http://localhost:8501`.

## 🧠 Model Training

The models are pre-trained and saved as `.pkl` artifacts. If you wish to retrain them on new data, you can run the respective training scripts:
- `python diabetes_model_training.py`
- `python heart_disease_model_training.py`
- `python breast_cancer_model_training.py`

*Note: The preprocessing scripts handle data scaling and feature selection automatically before training.*

## 🌐 Deployment

This project is fully deployment-ready for **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Click **New app**, select this repository, and set the main file path to `app.py`.
4. Click **Deploy**! Streamlit Cloud will automatically install the packages listed in `requirements.txt`.

## 📄 License

This project is open-source and available for educational and non-commercial clinical research purposes.
