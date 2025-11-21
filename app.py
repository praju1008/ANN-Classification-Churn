import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load model and preprocessors as before...
model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI (input widgets)
st.title("Customer Churn Prediction")
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ---- INSERT HERE ----
# Collect user input and encode/scale (PASTE CODE YOU POSTED HERE)
input_dict = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
}
input_df = pd.DataFrame(input_dict)

geo_array = onehot_encoder_geo.transform([[geography]])
geo_df = pd.DataFrame(
    geo_array,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

features_df = pd.concat([input_df, geo_df], axis=1)
features_scaled = scaler.transform(features_df)

prediction = model.predict(features_scaled)
prediction_prob = prediction[0][0]
# ---- END INSERT ----

# Show prediction to user
if prediction_prob > 0.5:
    st.error(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.success(f"The customer is unlikely to churn with a probability of {1 - prediction_prob:.2f}")
