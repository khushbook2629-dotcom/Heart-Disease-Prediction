import streamlit as st
import pandas as pd
import pickle

# Load Model
with open("Heart-Disease-Model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
numerical_cols = model_data["numerical_cols"]
categorical_cols = model_data["categorical_cols"]

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient details below.")

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input("Age", 20,100,50)

    Gender = st.selectbox(
        "Gender",
        [0,1]
    )

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        [0,1,2,3]
    )

    RestingBp = st.number_input(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

with col2:

    FastingBs = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    RestingECG = st.selectbox(
        "Resting ECG",
        [0,1,2]
    )

    MaxHR = st.number_input(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        [0,1]
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        [0,1,2]
    )

    MajorVessels = st.selectbox(
        "Major Vessels",
        [0,1,2,3,4]
    )

    Thalassemia = st.selectbox(
        "Thalassemia",
        [0,1,2,3]
    )

    ST_Depression = st.number_input(
        "ST Depression",
        0.0,
        10.0,
        1.0,
        step=0.1
    )


    input_df = pd.DataFrame({

    "Age":[Age],

    "Gender":[Gender],

    "ChestPainType":[ChestPainType],

    "RestingBp":[RestingBp],

    "Cholesterol":[Cholesterol],

    "FastingBs":[FastingBs],

    "RestingECG":[RestingECG],

    "MaxHR":[MaxHR],

    "ExerciseAngina":[ExerciseAngina],

    "ST_Slope":[ST_Slope],

    "MajorVessels":[MajorVessels],

    "Thalassemia":[Thalassemia],

    "ST_Depression":[ST_Depression]

})
    
    # One-Hot Encoding

input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_cols,
    drop_first=True
)

# Match feature names with training

input_encoded = input_encoded.reindex(
    columns=feature_names,
    fill_value=0
)

# Scale numerical columns

input_encoded[numerical_cols] = scaler.transform(
    input_encoded[numerical_cols]
)

if st.button("Predict Heart Disease"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    probability = float(probability)

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")

    st.subheader("Prediction Probability")
    st.progress(probability)
    st.write(f"Risk Score: {probability*100:.2f}%")

    st.caption(
        "Developed by Khushboo Kumari | Heart Disease Prediction System | 2026"
    )



