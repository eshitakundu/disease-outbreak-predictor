# Importing necessary libraries
import os
import pickle
import streamlit as st
import numpy as np

# Setting page configuration
st.set_page_config(
    page_title='Prediction of Disease Outbreaks',
    layout='centered',
    page_icon='🩺'
)

# Loading all models and scalers
def load_model_and_scaler(model_name):

    model_path = f"models/{model_name}.sav"
    scaler_path = f"models/{model_name}_scaler.pkl"

    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    return model, scaler

# Adding a background image
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://raw.githubusercontent.com/eshitakundu/disease-outbreak-predictor/main/background.png");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()

# Displaying Header
st.markdown("""
    <h1 style='text-align: center; color: rgb(153, 10, 10) ;'>Disease Prediction System</h1>
    <h4 style='text-align: center; color: rgb(58, 72, 86);'>Using Machine Learning Models</h4>
    <hr>
    """, unsafe_allow_html=True)

# Disease Prediction Option
option = st.radio("Choose a Prediction Type:", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Parkinson's Prediction"
], horizontal=True)

# Diabetes Prediction Page
if option == "Diabetes Prediction":
    st.subheader("Diabetes Prediction")

    # Load Diabetes model and scaler
    diabetes_model, diabetes_scaler = load_model_and_scaler("diabetes_model")

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)

    with col2:
        glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
        insulin = st.number_input("Insulin Level", min_value=0.0, step=0.1)
        age = st.number_input("Age", min_value=0, step=1)

    with col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)

    diab_diagnosis = ""

    if st.button("PREDICT"):
        try:
            # Convert user input into a NumPy array and ensure correct shape
            user_input = np.array([
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, diabetes_pedigree, age
            ]).reshape(1, -1) 

            # Scale the input
            user_input_scaled = diabetes_scaler.transform(user_input)

            # Make prediction
            prediction = diabetes_model.predict(user_input_scaled)  

            if prediction[0] == 1:
                st.error("The model predicts that the person is diabetic.")
            else:
                st.success("The model predicts that the person is not diabetic.")

        except Exception as e:
            st.error(f"Error in diabetes prediction: {e}")

# Heart Disease Prediction Page
elif option == "Heart Disease Prediction":
    st.subheader("Heart Disease Prediction")

    # Load Heart Disease model and scaler
    heart_model, heart_scaler = load_model_and_scaler("heart_disease_model")

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, step=1)
        sex = st.number_input("Sex (0: Female, 1: Male)", min_value=0, max_value=1, step=1)
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, step=1)
        chol = st.number_input("Cholesterol", min_value=0, step=1)
        fbs = st.number_input("Fasting Blood Sugar (1: >120 mg/dl, 0: otherwise)", min_value=0, max_value=1, step=1)
        thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3, step=1)

    with col2:
        restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, step=1)
        exang = st.number_input("Exercise-Induced Angina (1: Yes, 0: No)", min_value=0, max_value=1, step=1)
        oldpeak = st.number_input("ST Depression", step=0.1)
        slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, step=1)
        ca = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4, step=1)

    heart_diagnosis = ""

    if st.button("PREDICT"):
        try:
            user_input = np.array([age, sex, cp, trestbps, chol, fbs, 
                                    restecg, thalach, exang, oldpeak, 
                                    slope, ca, thal]).reshape(1,-1)
            user_input_scaled = heart_scaler.transform(user_input)
            prediction = heart_model.predict(user_input_scaled)

            if prediction[0] == 1:
                st.error("The model predicts that the person has heart disease.")
            else:
                st.success("The model predicts that the person does not have heart disease.")

        except Exception as e:
            st.error(f"Error in heart disease prediction: {e}")

# Parkinson's Disease Prediction Page
elif option == "Parkinson's Prediction":
    st.subheader("Parkinson's Disease Prediction")

    # Load Parkinson’s model and scaler
    parkinsons_model, parkinsons_scaler = load_model_and_scaler("parkinsons_model")

    # Input fields 

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, step=0.1)
        RAP = st.number_input('MDVP:RAP', min_value=0.0, step=0.1)
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, step=0.1)
        HNR = st.number_input('HNR', min_value=0.0, step=0.1)
        D2 = st.number_input('D2', min_value=0.0, step=0.1)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, step=0.1)
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, step=0.1)
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, step=0.1)
        RPDE = st.number_input('RPDE', min_value=0.0, step=0.1)
        PPE = st.number_input('PPE', min_value=0.0, step=0.1)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, step=0.1)
        DDP = st.number_input('Jitter:DDP', min_value=0.0, step=0.1)
        APQ = st.number_input('MDVP:APQ', min_value=0.0, step=0.1)
        DFA = st.number_input('DFA', min_value=0.0, step=0.1)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.1)
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, step=0.1)
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, step=0.1)
        spread1 = st.number_input('Spread1', min_value=0.0, step=0.1)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, step=0.1)
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, step=0.1)
        NHR = st.number_input('NHR', min_value=0.0, step=0.1)
        spread2 = st.number_input('Spread2', min_value=0.0, step=0.1)

    parkinsons_diagnosis = ""

    if st.button("PREDICT"):
        try:
            user_input = np.array([fo, fhi, flo, Jitter_percent, Jitter_Abs,
                                    RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                                    APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]).reshape(1,-1)
            user_input_scaled = parkinsons_scaler.transform(user_input)
            prediction = parkinsons_model.predict(user_input_scaled)

            if prediction[0] == 1:
                st.error("The model predicts Parkinson's disease.")
            else:
                st.success("The model predicts no Parkinson's disease.")

        except Exception as e:
            st.error(f"Error in Parkinson's prediction: {e}")

st.error("This system uses machine learning models. Consult a healthcare professional for an official diagnosis.")