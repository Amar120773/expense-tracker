# app/app.py

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="Expense Categorizer", layout="centered")

st.title("💰 Smart Expense Categorization System")
st.write("Enter your expense and get category prediction")

# =========================
# User Input
# =========================
description = st.text_input("Enter expense description")
amount = st.number_input("Enter amount", min_value=0)

# =========================
# Prediction
# =========================
if st.button("Predict Category"):
    if description.strip() == "":
        st.warning("Please enter a description")
    else:
        vec = vectorizer.transform([description.lower()])
        prediction = model.predict(vec)[0]
        st.success(f"Predicted Category: {prediction}")

# =========================
# CSV Upload for Insights
# =========================
st.subheader("📊 Upload CSV for Expense Insights")

uploaded_file = st.file_uploader("Upload your expenses CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:")
    st.dataframe(df.head())

    if "category" in df.columns and "amount" in df.columns:
        category_sum = df.groupby("category")["amount"].sum()

        st.subheader("Expense Distribution")

        fig, ax = plt.subplots()
        ax.pie(category_sum, labels=category_sum.index, autopct="%1.1f%%")
        ax.set_title("Spending by Category")

        st.pyplot(fig)
    else:
        st.error("CSV must contain 'category' and 'amount' columns")