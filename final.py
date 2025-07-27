import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

print("🔄 Loading dataset...")
try:
    df = pd.read_csv("crop_dataset.csv")
    print(f"✅ Dataset loaded. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit()

print("🔧 Cleaning data...")
df.dropna(subset=['Production'], inplace=True)
df.fillna(0, inplace=True)

print("🧠 Encoding categorical features...")
label_encoders = {}
for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

print("📊 Preparing features and target...")
X = df[['State_Name_encoded', 'District_Name_encoded', 'Season_encoded', 'Crop_encoded', 'Area']]
y = df['Production']

print("🧪 Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("💾 Saving model and encoders...")
joblib.dump(model, "crop_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model trained and saved successfully.")
