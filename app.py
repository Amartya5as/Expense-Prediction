import streamlit as st
import pandas as pd
import joblib
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

model = joblib.load("expense_classifier_pipeline.pkl")

# Manually define category mapping (update this if you have the real one)
category_map = {
    0: "Groceries",
    1: "Utilities",
    2: "Entertainment",
    3: "Transport",
    4: "Healthcare"
}

st.title("💸 Expense Category Predictor")

# User input form
with st.form("predict_form"):
    transaction_description = st.text_input("Transaction Description", "Spotify")
    amount = st.number_input("Amount", min_value=0.0, value=199.0)
    month = st.selectbox("Month", list(range(1, 13)), index=9)
    day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)), index=2)
    submitted = st.form_submit_button("Predict")

# Make prediction
if submitted:
    input_df = pd.DataFrame([{
        "transaction_description": transaction_description,
        "amount": amount,
        "month": month,
        "day_of_week": day_of_week
    }])
    
    prediction = model.predict(input_df)
    category_id = prediction[0]
    category_name = category_map.get(category_id, "Unknown")

    st.success(f"🧾 Predicted Category: **{category_name}**")
