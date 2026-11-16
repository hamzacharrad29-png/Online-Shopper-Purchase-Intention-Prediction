"""Streamlit app: predict whether an online shopping session ends with a purchase.

Run the notebook first (it saves the model files to models/), then:
    streamlit run app.py
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent / "models"


@st.cache_resource
def load(name):
    with open(MODELS / name, "rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="Online Shoppers Purchase Intention", layout="centered")
st.title("Will the visitor make a purchase?")

if not (MODELS / "decision_tree_model.pkl").exists():
    st.error("Model files not found. Run notebooks/purchase_intention_prediction.ipynb first: "
             "it saves the model and preprocessing objects to models/.")
    st.stop()

model = load("decision_tree_model.pkl")
scaler = load("scaler.pkl")
encoder = load("onehot_encoder.pkl")
feature_columns = load("feature_columns.pkl")

st.markdown("Enter the session details below.")
col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input("Administrative pages visited", min_value=0, step=1)
    administrative_duration = st.number_input("Time on administrative pages (seconds)", min_value=0.0, step=1.0)
    informational = st.number_input("Informational pages visited", min_value=0, step=1)
    informational_duration = st.number_input("Time on informational pages (seconds)", min_value=0.0, step=1.0)
    product_related = st.number_input("Product-related pages visited", min_value=0, step=1)
    product_related_duration = st.number_input("Time on product pages (seconds)", min_value=0.0, step=1.0)
    bounce_rates = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.01)
    exit_rates = st.slider("Exit rate", 0.0, 0.2, 0.04, 0.01)
    page_values = st.number_input("Page value", min_value=0.0, step=1.0)

with col2:
    special_day = st.slider("Special day (closeness to a holiday)", 0.0, 1.0, 0.0, 0.05)
    operating_systems = st.selectbox("Operating system", options=list(range(1, 9)), index=1)
    browser = st.selectbox("Browser", options=list(range(1, 14)), index=1)
    region = st.selectbox("Region", options=list(range(1, 10)), index=1)
    traffic_type = st.selectbox("Traffic type", options=list(range(1, 21)), index=1)
    weekend = st.selectbox("Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    month = st.selectbox("Month", options=["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    visitor_type = st.selectbox("Visitor type", options=["Returning_Visitor", "New_Visitor", "Other"])

# Build the feature vector exactly as in training
input_df = pd.DataFrame([{
    "Administrative": administrative,
    "Administrative_Duration": administrative_duration,
    "Informational": informational,
    "Informational_Duration": informational_duration,
    "ProductRelated": product_related,
    "ProductRelated_Duration": product_related_duration,
    "BounceRates": bounce_rates,
    "ExitRates": exit_rates,
    "PageValues": page_values,
    "SpecialDay": special_day,
    "OperatingSystems": operating_systems,
    "Browser": browser,
    "Region": region,
    "TrafficType": traffic_type,
    "Weekend": weekend,
}])
encoded = encoder.transform(pd.DataFrame([[month, visitor_type]], columns=encoder.feature_names_in_))
input_df = pd.concat([input_df, pd.DataFrame(encoded, columns=encoder.get_feature_names_out())], axis=1)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

if st.button("Predict purchase intention"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.success(f"The visitor is likely to purchase (probability: {proba:.2f}).")
    else:
        st.error(f"The visitor is unlikely to purchase (probability: {proba:.2f}).")
