import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("premium_model.pkl")

st.title("Health Insurance Premium Estimator")

# User Inputs
age = st.number_input("Enter your Age", min_value=18, max_value=100, step=1)
height = st.number_input("Enter your Height (cm)", min_value=100, max_value=250)
weight = st.number_input("Enter your Weight (kg)", min_value=30, max_value=200)
diabetes = st.radio("Do you have Diabetes?", ["No", "Yes"])
bp_problems = st.radio("Do you have Blood Pressure Problems?", ["No", "Yes"])

# Convert inputs into DataFrame
user_data = pd.DataFrame({
    'Age': [age],
    'Diabetes': [1 if diabetes == "Yes" else 0],
    'BloodPressureProblems': [1 if bp_problems == "Yes" else 0],
    'Height': [height],
    'Weight': [weight]
})

# Predict premium when button is clicked
if st.button("Estimate Premium"):
    premium_pred = model.predict(user_data)[0]
    formatted_premium = f"₹ {round(premium_pred):,}"
    st.success(f"Estimated Premium for 10,00,000 coverage amount is: {formatted_premium}")
