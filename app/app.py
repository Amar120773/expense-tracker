import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Expense AI", layout="wide")

# =========================
# Load Model
# =========================
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# =========================
# Custom CSS
# =========================
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f2f2f2;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("🔧 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📊 Insights"])

# =========================
# HOME PAGE
# =========================
if option == "🏠 Home":

    st.markdown('<p class="main-title">💰 Smart Expense Categorization</p>', unsafe_allow_html=True)
    st.write("AI-powered system to classify your expenses instantly")

    st.markdown("### 🧾 Enter Expense Details")

    col1, col2 = st.columns(2)

    with col1:
        description = st.text_input("Enter description", placeholder="e.g., pizza order")

    with col2:
        amount = st.number_input("Enter amount", min_value=0)

    if st.button("🚀 Predict Category"):

        if description.strip() == "":
            st.warning("⚠ Please enter a description")
        else:
            vec = vectorizer.transform([description.lower()])
            prediction = model.predict(vec)[0]

            st.markdown(f"""
                <div class="card">
                    <h3>✅ Predicted Category: <span style='color:green'>{prediction.upper()}</span></h3>
                    <p>💵 Amount: ₹{amount}</p>
                </div>
            """, unsafe_allow_html=True)

# =========================
# INSIGHTS PAGE
# =========================
elif option == "📊 Insights":

    st.title("📊 Expense Insights Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if "category" in df.columns and "amount" in df.columns:

            category_sum = df.groupby("category")["amount"].sum()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥧 Category Distribution")
                fig, ax = plt.subplots()
                ax.pie(category_sum, labels=category_sum.index, autopct="%1.1f%%")
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Bar Chart")
                st.bar_chart(category_sum)

        else:
            st.error("CSV must contain 'category' and 'amount' columns")