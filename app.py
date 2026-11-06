import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Retail Sales Forecasting")

# Find project files
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

model_path = next(
    (p for p in [
        BASE_DIR / "best_retail_model.pkl",
        PROJECT_DIR / "best_retail_model.pkl"
    ] if p.exists()),
    None
)

csv_path = next(
    (p for p in [
        BASE_DIR / "synthetic_online_retail_data.csv",
        PROJECT_DIR / "synthetic_online_retail_data.csv"
    ] if p.exists()),
    None
)

if model_path is None:
    st.error("Model file not found! Check best_retail_model.pkl location.")
    st.stop()

if csv_path is None:
    st.error("Dataset CSV file not found!")
    st.stop()

# Load model and dataset
model = joblib.load(model_path)
df = pd.read_csv(csv_path)

st.subheader("Sales Dashboard")
st.metric("Total Orders", len(df))
st.metric("Total Quantity Sold", int(df["quantity"].sum()))

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Predict Sales Quantity")

inputs = {}

for col in df.columns:
    if col not in ["quantity", "order_date"]:
        if pd.api.types.is_numeric_dtype(df[col]):
            inputs[col] = st.number_input(col, value=0.0)
        else:
            options = df[col].dropna().astype(str).unique().tolist()
            inputs[col] = st.selectbox(col, options)

if st.button("Predict"):
    data = pd.DataFrame([inputs])
    data = pd.get_dummies(data, drop_first=True)

    if hasattr(model, "feature_names_in_"):
        data = data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    prediction = model.predict(data)[0]

    st.success(
        f"Predicted Quantity: {max(0, prediction):.2f}"
    )