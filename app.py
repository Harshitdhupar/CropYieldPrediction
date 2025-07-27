import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("crop_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
df = pd.read_csv("crop_dataset.csv")

st.set_page_config(page_title="Crop Production Predictor", layout="centered")
st.title("🌾 Crop Production Prediction App")

# Select inputs
state = st.selectbox("Select State", sorted(df['State_Name'].unique()))
districts = df[df['State_Name'] == state]['District_Name'].unique()
district = st.selectbox("Select District", sorted(districts))
season = st.selectbox("Select Season", sorted(df['Season'].unique()))
crop = st.selectbox("Select Crop", sorted(df['Crop'].unique()))
area = st.number_input("Enter Area (in hectares)", min_value=0.1)

if st.button("Predict Production"):
    try:
        # Encode user input
        state_enc = label_encoders['State_Name'].transform([state])[0]
        district_enc = label_encoders['District_Name'].transform([district])[0]
        season_enc = label_encoders['Season'].transform([season])[0]
        crop_enc = label_encoders['Crop'].transform([crop])[0]

        input_data = np.array([[state_enc, district_enc, season_enc, crop_enc, area]])
        prediction = model.predict(input_data)

        st.success(f"🌾 Estimated Production: {prediction[0]:.2f} metric tons")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
