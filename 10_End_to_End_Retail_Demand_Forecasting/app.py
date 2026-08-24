import pandas as pd
import numpy as np
import streamlit as st
import pickle

@st.cache_resource
def load_artifact():
    with open("xgboost_demand_model_RSCV.pkl", "rb") as file:
        model = pickle.load(file)
    
    with open("label_encoder.pkl", "rb") as file:
        encoders = pickle.load(file)

    return model, encoders 

model, label_encoders = load_artifact()



st.title("Demand Forcasting App")

st.divider()

st.header("Input Features")


price = st.number_input("Price", min_value= 0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value = 0, max_value = 100, value = 10)
inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", [0, 1])
compotitor_pricing = st.number_input("Compotitor Price", min_value=0.0, value= 50.0)
category = st.selectbox("Category", label_encoders.classes_.tolist())

input_data = pd.DataFrame({
    "Price" : [price],
    "Competitor Pricing" : [compotitor_pricing],
    "Discount" : [discount],
    "Inventory Level" : [inventory_level],
    "Category" : [category],
    "Promotion" : [promotion]
})



input_data["Category"] = label_encoders.transform(input_data["Category"])


st.divider()

if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction Demand: {int(prediction)} Units")
