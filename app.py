
import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction Using Machine Learning | Saurabh Dahake",
    page_icon="🩺",
    layout="centered"
)

st.title("Diabetes Prediction Using Machine Learning")
st.header("Diabetes Prediction by Saurabh Dahake")
st.write(
    "A Machine Learning web application "
    "for diabetes prediction using Python."
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Diabetes Prediction App")
st.write("Enter the medical details below to check the prediction.")

# Input fields for the 8 features matching the model training data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=79.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare input data array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.subheader("Result")
    if prediction[0] == 1:
        st.error("The model predicts that the individual is **diabetic**.")
    else:
        st.success("The model predicts that the individual is **not diabetic**.")
