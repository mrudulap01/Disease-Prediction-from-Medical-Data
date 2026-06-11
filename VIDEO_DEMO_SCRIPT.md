# Video Demonstration Script: MediPredict AI

**Estimated Duration:** 2.5 - 3 Minutes

---

## 1. Introduction (0:00 - 0:30)
**[Visuals]**
Start with the camera on you (the presenter) or displaying the project title slide. Transition to the MediPredict AI Dashboard landing page.

**[Audio]**
"Hello everyone! Welcome to my internship project presentation. Today, I'm thrilled to showcase **MediPredict AI**, a comprehensive machine-learning platform I built to assist in the early detection and diagnosis of chronic diseases. The goal of this project was to take complex clinical data and transform it into an accessible, instantly actionable tool for both patients and healthcare professionals."

## 2. Platform Overview & UI (0:30 - 1:00)
**[Visuals]**
Navigate around the app. Scroll down the patient form. Show the "Glassmorphism" UI, highlight the input sliders, and point out the "Quick Test Profiles" buttons.

**[Audio]**
"As you can see, the platform features a modern, premium user interface. It’s entirely web-based, built using Python and Streamlit. The application currently supports three distinct predictive models: Diabetes, Heart Disease, and Breast Cancer. Let’s take a look at the Diabetes model. The user can manually input their clinical vitals—like Glucose levels, Blood Pressure, and BMI—using these interactive sliders. However, for the sake of demonstration, I’ve built in these 'Quick Test Profile' buttons that allow us to instantly load sample data."

## 3. Disease Selection & Prediction Demo (1:00 - 2:00)
**[Visuals]**
1. Click the "High Risk" button under Diabetes.
2. Click "Generate AI Diagnostic Report".
3. Show the loading spinner, then reveal the red "High Risk" card.
4. Switch the sidebar dropdown to "Heart Disease" (show how the UI repaints dynamically).
5. Click the "Random Patient" or "Healthy" button.
6. Click "Generate AI Diagnostic Report" and show the green "Low Risk" card.

**[Audio]**
"Let’s run a test. I'll load a 'High Risk' diabetic profile. Once I hit 'Generate', the backend passes this data through a trained Random Forest Classifier. Instantly, we get a diagnostic report with a confidence score. The model accurately flags this as a High Risk scenario. 

What makes this platform truly powerful is its dynamic nature. If I switch the sidebar selection to Heart Disease, the entire form instantly repaints to request cardiovascular metrics—such as Cholesterol and Max Heart Rate. Let's test a Healthy profile this time... and as expected, the model clears the patient with a Low Risk status."

## 4. PDF Generation & Analytics (2:00 - 2:30)
**[Visuals]**
1. Click the "Download PDF Report" button.
2. Open the downloaded PDF on screen.
3. Switch the sidebar navigation from "Dashboard" to "History".
4. Show the prediction history table.

**[Audio]**
"In a real clinical setting, documentation is vital. Therefore, I engineered a feature that automatically compiles the patient's inputs and the AI's prediction into a professional, downloadable PDF report with just one click. Furthermore, the platform maintains a live session history log, allowing doctors to track multiple assessments efficiently."

## 5. Conclusion (2:30 - 3:00)
**[Visuals]**
Return to the main Dashboard or show a slide with the ML performance table.

**[Audio]**
"Under the hood, these models were rigorously trained using Scikit-Learn on standard medical datasets, achieving over 93% cross-validated accuracy. This project has been an incredible journey, allowing me to bridge the gap between complex data science algorithms and practical, user-friendly software engineering. Thank you for watching my demonstration!"
