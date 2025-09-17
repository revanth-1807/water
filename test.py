import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 1. Load model and preprocessing tools
model = load_model("best_model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))
le_quality = pickle.load(open("le_quality.pkl", "rb"))
le_diseases = pickle.load(open("le_diseases.pkl", "rb"))

# 2. Test data
test_data = pd.DataFrame([
    [7.2, 180.0, 15000.0, 7.5, 320.0, 400.0, 12.0, 50.0, 3.5],
    [6.8, 200.0, 25000.0, 5.0, 300.0, 450.0, 15.0, 70.0, 4.0],
    [8.5, 250.0, 50000.0, 2.0, 500.0, 1000.0, 25.0, 150.0, 8.0],
    [5.5, 100.0, 70000.0, 1.0, 600.0, 1200.0, 30.0, 200.0, 9.0]
], columns=[
    "ph","Hardness","Solids","Chloramines","Sulfate",
    "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
])

# 3. Scale test data
X_test = scaler.transform(test_data)

# 4. Predictions
yq_pred, yd_pred = model.predict(X_test)
yq_labels = np.argmax(yq_pred, axis=1)
yd_labels = np.argmax(yd_pred, axis=1)

# 5. Decode predictions
pred_quality = le_quality.inverse_transform(yq_labels)
pred_diseases = le_diseases.inverse_transform(yd_labels)

# 6. Map long disease text to short names
disease_map = {
    "Unknown contamination (no specific indicator flagged)": "Unknown",
    "None (potable)": "None",
    "Indicates dissolved solids — possible gastrointestinal issues; Laxative effects / diarrhea at high concentrations; Long-term exposure linked to liver/kidney and cancer risks; short-term: irritation": "High Solids Risk"
}

pred_diseases_short = [disease_map.get(d, d) for d in pred_diseases]

# 7. Clean DataFrame output
predictions = pd.DataFrame({
    "Predicted_Quality": pred_quality,
    "Predicted_Diseases": pred_diseases_short
})

# 8. Print results
print(predictions)
