import streamlit as st
import requests

# Streamlit UI
st.title("Machine Learning Prediction")

yellow = st.number_input("Age of the student", min_value=15, max_value=18)
red = st.selectbox("Gender of the student", [0, 1])
position_encoded = st.selectbox("Ethnicity", [0, 1, 2, 3])

if st.button("Predict"):
    response = requests.get("http://localhost:8000/predict", params={"yellow": yellow, "red": red, "position_encoded": position_encoded})
    
    if response.status_code == 200:
        predictions = response.json()
        st.write("Predictions:", predictions)
    else:
        st.error("Prediction failed. Please check your inputs.")