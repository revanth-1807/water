
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load scaler and label encoders
scaler = pickle.load(open("scaler.pkl", "rb"))
le_quality = pickle.load(open("le_quality.pkl", "rb"))
le_diseases = pickle.load(open("le_diseases.pkl", "rb"))

# Load trained model
model = tf.keras.models.load_model("best_model.keras")

st.title("💧 Water Quality & Disease Prediction Dashboard")

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0)
solids = st.number_input("Solids", min_value=0.0)
chloramines = st.number_input("Chloramines", min_value=0.0)
sulfate = st.number_input("Sulfate", min_value=0.0)
conductivity = st.number_input("Conductivity", min_value=0.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
turbidity = st.number_input("Turbidity", min_value=0.0)

if st.button("🔮 Predict"):
    # Prepare input
    sample = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])
    sample_scaled = scaler.transform(sample)

    # Predict
    pred_quality, pred_diseases = model.predict(sample_scaled)

    predicted_quality = le_quality.inverse_transform([np.argmax(pred_quality)])
    predicted_disease = le_diseases.inverse_transform([np.argmax(pred_diseases)])

    st.success(f"✅ Predicted Quality: {predicted_quality[0]}")
    st.success(f"✅ Predicted Disease: {predicted_disease[0]}")
