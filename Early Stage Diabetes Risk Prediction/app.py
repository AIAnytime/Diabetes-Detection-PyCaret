import streamlit as st 
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 

from pycaret.classification import load_model, predict_model

#------------------------------------------------

#load our saved model
model = load_model("mlmodel")

#---------------------------
# function to load our data
@st.cache 
def load_data(data):
    df = pd.read_csv(data)
    return df 

# function to make prediction
@st.cache
def predict(model, input_df):
    prediction_df = predict_model(estimator = model, data= input_df)
    prediction = prediction_df['Label'][0]
    return prediction

#---------------------------------------------------

# main function to run the app
def run():
    st.title("Early Stage Diabetes Risk Prediction")

    age_var = st.number_input("Age", min_value=20, max_value = 80, value = 40)
    gender_var = st.radio("Gender", ["Male", "Female"])
    polyuria_var = st.radio("Polyuria", ["Yes", "No"])
    polydipsia_var = st.radio("Polydipsia", ["Yes", "No"])
    sudden_weight_loss_var = st.radio("Sudden Weight Loss", ["Yes", "No"])
    weakness_var = st.radio("Any Weakness", ["Yes", "No"])
    polyphagia_var = st.radio("Polyphagia", ["Yes", "No"])
    genital_thrush_var = st.radio("Genital Thrush", ["Yes", "No"])
    visual_blurring_var = st.radio("Visual Blurring", ["Yes", "No"])
    itching_var = st.radio("Itching", ["Yes", "No"])
    irritability_var = st.radio("Any Irritation", ["Yes", "No"])
    delayed_healing_var = st.radio("Delayed Healing", ["Yes", "No"])
    partial_paresis_var = st.radio("Partial Paresis", ["Yes", "No"])
    muscle_stiffness_var = st.radio("Muscle Stiffness", ["Yes", "No"])
    alopecia_var = st.radio("Alopecia", ["Yes", "No"])
    obesity_var = st.radio("Obesity", ["Yes", "No"])

    input_dict = {
        "Age":age_var,
        "Gender":gender_var,
        "Polyuria":polyuria_var,
        "Polydipsia":polydipsia_var,
        "sudden weight loss":sudden_weight_loss_var,
        "weakness":weakness_var,
        "Polyphagia":polyphagia_var,
        "Genital thrush":genital_thrush_var,
        "visual blurring":visual_blurring_var,
        "Itching":itching_var,
        "Irritability":irritability_var,
        "delayed healing":delayed_healing_var,
        "partial paresis":partial_paresis_var,
        "muscle stiffness":muscle_stiffness_var,
        "Alopecia":alopecia_var,
        "Obesity":obesity_var
    }

    input_data = pd.DataFrame([input_dict])
    st.table(input_data)

    if st.button("Predict"):
        output_result = predict(model, input_data)
        if output_result == "Negative":
            st.write("Hey, No signs of Diabetes!!")
        elif output_result == "Positive":
            st.write("Risk of Diabetes. Consult a Doctor!!")

if __name__ == '__main__':
    run()
