import streamlit as st
import pandas as pd
import joblib
import os


st.title("🍷 Wine Quality Classifier")
st.write("Enter chemical properties of red wine to predict if it's likely a **good wine**.")

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"

missing_files = [p for p in [MODEL_PATH, SCALER_PATH] if not os.path.exists(p)]
if missing_files:
    st.error(f"Missing model or preprocessing files: {', '.join(missing_files)}")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with st.form("wine_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
        chlorides = st.number_input("Chlorides", min_value=0.0, step=0.01)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)

    with col2:
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
        density = st.number_input("Density", min_value=0.0, step=0.001)

    with col3:
        pH = st.number_input("pH", min_value=0.0, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

    submit = st.form_submit_button("Predict Quality")

if submit:
    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]], columns=[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ])

    if (input_data == 0).all(axis=1).iloc[0]:
        st.error("🚫 All input features are zero. Please enter valid wine characteristics to proceed.")
    else:
        X_scaled = scaler.transform(input_data)
        X_scaled_df = pd.DataFrame(X_scaled, columns=input_data.columns)

        probability = model.predict_proba(X_scaled_df)[0][1] 
        threshold = 0.6
        prediction = int(probability >= threshold)

        if prediction == 1:
            st.success("✅ This wine is predicted to be **GOOD**")
            st.success(f"🧠 Confidence Score: **{probability:.2%}** of being GOOD")
        else:
            st.warning("⚠️ This wine is predicted to be **NOT GOOD**")
            st.warning(f"🧠 Confidence Score: **{1 - probability:.2%}** of being NOT GOOD")

