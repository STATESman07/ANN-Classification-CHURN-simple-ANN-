import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

## load the trained model,
model = load_model("model.keras")

# Load the label encoder, one-hot encoder and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("One_Hot_Encoder_geo.pkl", "rb") as file:
    One_Hot_Encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

print(One_Hot_Encoder_geo.categories_)
print(label_encoder_gender.classes_)
## streamit.app
st.title("Customer Churn Prediction")

# Input features
geography = st.selectbox("Geography", One_Hot_Encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("BALANCE")
credit_score = st.number_input("CREDIT SCORE")
estimated_salary = st.number_input("ESTIMATED SALARY")
tenure = st.slider("TENURE", 0, 10)
number_of_products = st.slider("NUMBER OF PRODUCTS", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has_Cr_Card", [0, 1])
is_active_memeber = st.selectbox("Is_Active_Member", [0, 1])

# prepare the input_data

input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [number_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_memeber],
        "EstimatedSalary": [estimated_salary],
    }
)


# one_hot encode "GEOGRAPHY"
geo_encoder = One_Hot_Encoder_geo.transform([[geography]])
geo_encoder_df = pd.DataFrame(
    geo_encoder, columns=One_Hot_Encoder_geo.get_feature_names_out(["Geography"])
)

# concatination of one hot encode_Df
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)


# scale the input data
input_scaler = scaler.transform(input_data)

# prediction
prediction = model.predict(input_scaler)
prediction_proba = prediction[0][0]
print(prediction_proba, sep="----->[0][1]")


if prediction_proba > 0.5:
    prediction_is = "Churn"
else:
    prediction_is = "Not Churn"

st.write(f"Prediction: {prediction}")
st.write(f"Prediction_proba: {prediction_proba:.2f}")
st.write(f"Predictionis : {prediction_is}")
