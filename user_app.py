import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import logging

# Logging File
logging.basicConfig(
    filename='app.log',        # Log file name
    level=logging.INFO,        # Minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load trained model and dataset
model = joblib.load("diabetes_model.pkl")
df = pd.read_csv("diabetes.csv")

# Split data for validation (confusion matrix later)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# page navigatiing
if 'page' not in st.session_state:
    st.session_state['page'] = "Home"

# Moves to results page if the user has pressed submit
if 'submitted' in st.session_state and st.session_state['submitted']:
    st.session_state['page'] = "Result"

# Sidebar
page = st.sidebar.selectbox("Navigation", ["Home", "Result"], index=["Home", "Result"].index(st.session_state['page']))


# Home Page
if page == "Home":
    st.title("Diabetes Prediction App")
    st.header("Enter Patient Information")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=21, max_value=120, value=33)

    if st.button("Submit"):
        st.session_state['user_input'] = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                                    insulin, bmi, dpf, age]])
        st.session_state['submitted'] = True
        st.success("Input submitted! Redirecting to Result page.")
        st.session_state['page'] = "Result"
        st.rerun()


# Result Page
elif page == "Result":
    st.title("Diabetes Prediction Result")

    if 'submitted' in st.session_state and st.session_state['submitted']:
        input_data = pd.DataFrame(st.session_state['user_input'], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])

        # Prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("The model predicts: Diabetic")
        else:
            st.success("The model predicts: Not Diabetic")

        # Confusion Matrix
        st.subheader("Model Evaluation: Confusion Matrix")

        val_preds = model.predict(X_val)
        cm = confusion_matrix(y_val, val_preds)
        cm_percent = cm.astype('float') / cm.sum() * 100 # turning value into percentages
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["Not Diabetic", "Diabetic"])

        fig, ax = plt.subplots(figsize=(5,5))
        disp.plot(ax=ax, values_format=".2f")
        st.pyplot(fig)


        # Glucose Histogram
        st.subheader("Glucose Level Comparison")

        fig2, ax2 = plt.subplots()
        ax2.hist(df[df['Outcome'] == 0]['Glucose'], bins=20, alpha=0.5, label='Not Diabetic')
        ax2.hist(df[df['Outcome'] == 1]['Glucose'], bins=20, alpha=0.5, label='Diabetic')
        ax2.axvline(input_data['Glucose'][0], color='red', linestyle='dashed', linewidth=2, label='Patient Glucose')
        ax2.set_xlabel('Glucose Level')
        ax2.set_ylabel('Number of Patients')
        ax2.legend()
        st.pyplot(fig2)

        # BMI Histogram
        st.subheader("BMI Comparison")

        fig3, ax3 = plt.subplots()
        ax3.hist(df[df['Outcome'] == 0]['BMI'], bins=20, alpha=0.5, label='Not Diabetic')
        ax3.hist(df[df['Outcome'] == 1]['BMI'], bins=20, alpha=0.5, label='Diabetic')
        ax3.axvline(input_data['BMI'][0], color='red', linestyle='dashed', linewidth=2, label='Patient BMI')
        ax3.set_xlabel('BMI')
        ax3.set_ylabel('Number of Patients')
        ax3.legend()
        st.pyplot(fig3)


        # Feature Importance
        st.subheader("Feature Importance (Model Insights)")

        importances = model.feature_importances_
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=True)

        fig4, ax4 = plt.subplots()
        ax4.barh(importance_df['Feature'], importance_df['Importance'])
        ax4.set_xlabel('Importance Score')
        ax4.set_title('Feature Importance (higher = more important)')
        st.pyplot(fig4)

    else:
        st.warning("No input submitted yet. Please go to the 'Home' page first.")
