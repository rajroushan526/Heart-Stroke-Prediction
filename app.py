import streamlit as st
import pandas as pd 
import joblib

model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# st.title("Heart Stroke Prediction BY ROSHAN")
st.markdown("<h1 style='text-align: center;'>Heart Stroke Prediction By ROSHAN❤️</h1>", unsafe_allow_html=True)
st.markdown("Provide the following details")

age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX",['Male', 'Female'])

chest_pain = st.selectbox("Chest Pain Type", ["ATA" , "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)", 80,200, 100)
Cholesterol = st.number_input("Cholesterol(mg/dl)" , 100,600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60,220,140)
exercise_angina = st.selectbox("Excercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0,6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat","Down"])



if st.button("Predict"):
    raw_input ={
        'Age' : age,
        'RestingBP': resting_bp,
        'Cholesterol': Cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'OldPeak':oldpeak,
        'Sex': sex,
        'ChestPain'+ chest_pain : 1,
        'RestingECG'+ resting_ecg : 1,
        'ExerciseAngina_' + exercise_angina : 1,
        'ST_Slope_' + st_slope : 1


    }


    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    if prediction == 1:
        st.error("⚠️ High Risk OF HEART DISEASE")
    else:
        st.success("✅Low Risk OF HEART DISEASE")