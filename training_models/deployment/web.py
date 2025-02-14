#Import necessary libraries
import os
import pickle
import streamlit as st

#Setting page configuration
st.set_page_config(
    page_title='Prediction of Disease Outbreaks',
    layout='centered',
    page_icon="🩺"
)

working_dir = os.path.dirname(__file__)

#Loading saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav','rb'))

#adding a background image
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()

#Displaying Header
st.markdown("""
    <h1 style='text-align: center; color:rgb(153, 10, 10) ;'>Disease Prediction System</h1>
    <h4 style='text-align: center; color:rgb(58, 72, 86);'>Using Machine Learning Models</h4>
    <hr>
    """, unsafe_allow_html=True)

#Disease Prediction Option
option = st.radio("Choose a Prediction Type:", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Parkinson's Prediction"
], horizontal=True)

#Diabetes Prediction Page
if option == 'Diabetes Prediction':
    st.subheader('Diabetes Prediction')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        SkinThickness = st.number_input('Skin Thickness Value', min_value=0.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0, step=0.1)

    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, step=0.1)
        Insulin = st.number_input('Insulin Level', min_value=0.0, step=0.1)
        Age = st.number_input('Age of the Person', min_value=0, step=1)

    with col3:
        Bloodpressure = st.number_input('Blood Pressure Value', min_value=0.0, step=0.1)
        BMI = st.number_input('BMI Value', min_value=0.0, step=0.1)

    diab_diagnosis = ''

    if st.button('PREDICT'):
        try:
            user_input = [
                Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age
            ]
            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except Exception as e:
            st.error(f"Error in diabetes prediction: {e}")

    st.success(diab_diagnosis)

#Heart Disease Prediction Page
elif option == 'Heart Disease Prediction':
    st.subheader("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=0, step=1)
        sex = st.number_input('Sex (0: Female, 1: Male)', min_value=0, max_value=1, step=1)
        cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, step=1)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, step=1)
        chol = st.number_input('Cholesterol Level', min_value=0, step=1)
        fbs = st.number_input('Fasting Blood Sugar', min_value=0, max_value=1, step=1)
        thal = st.number_input('Thalassemia', min_value=0, step=1)

    with col2:
        restecg = st.number_input('ECG Results', step=1)
        thalach = st.number_input('Max Heart Rate Achieved', min_value=0, step=1)
        exang = st.number_input('Exercise-Induced Angina (1: Yes, 0: No)', min_value=0, max_value=1, step=1)
        oldpeak = st.number_input('ST Depression', step=0.1)
        slope = st.number_input('ST Slope', step=0.1)
        ca = st.number_input('Major Vessels', step=1)

    heart_diagnosis = ''

    if st.button('PREDICT'):
        try:
            user_input = [
                float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang), float(oldpeak),
                float(slope), float(ca), float(thal)
            ]
            user_input = [float(x) for x in user_input]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'
        except Exception as e:
            st.error(f"Error in heart disease prediction: {e}")

    st.success(heart_diagnosis)

#Parkinson's Disease Prediction Page
if option == "Parkinson's Prediction":
    st.subheader("Parkinson's Disease Prediction")

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
            user_input = [
                fo, fhi, flo, Jitter_percent, Jitter_Abs,
                RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
            ]
            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except Exception as e:
            st.error(f"Error in Parkinson's prediction: {e}")

    st.success(parkinsons_diagnosis)

st.info("This system uses machine learning models. Consult a healthcare professional for an official diagnosis.")
