import pandas as pd
import numpy as np

input_path = "water_potability.csv"
output_path = "water_potability2.csv"

df = pd.read_csv(input_path)

# Fill missing values with median for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Potability" in numeric_cols:
    numeric_cols.remove("Potability")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ---------------------------
# Define Safe Ranges (WHO/BIS)
# ---------------------------
safe_ranges = {
    "ph": (6.5, 8.5),
    "Hardness": (50, 150),
    "Solids": (0, 1000),  # typical acceptable up to 1000
    "Chloramines": (0.2, 2),
    "Sulfate": (0, 250),
    "Conductivity": (50, 500),
    "Organic_carbon": (0, 10),
    "Trihalomethanes": (0, 80),
    "Turbidity": (0, 5),
}

# ---------------------------
# Disease Mapping
# ---------------------------
disease_map = {
    "ph": "Skin/eye irritation; Gastrointestinal upset",
    "Hardness": "Taste/mineral effects; possible kidney stone risk",
    "Solids": "Gastrointestinal illness (diarrhea, stomach upset)",
    "Chloramines": "Low: bacterial risk; High: irritation/chemical upset",
    "Sulfate": "Laxative effects / diarrhea at high concentrations",
    "Conductivity": "Indicates dissolved solids — possible gastrointestinal issues",
    "Organic_carbon": "Can form harmful byproducts during disinfection",
    "Trihalomethanes": "Long-term: liver/kidney issues, cancer risk",
    "Turbidity": "Waterborne pathogens — diarrhea, cholera, typhoid risk",
}

# ---------------------------
# Disease Assignment Function
# ---------------------------
def assign_diseases(row):
    issues = []
    for col, (low, high) in safe_ranges.items():
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if not (low <= val <= high):
            if col in disease_map:
                issues.append(disease_map[col])
            else:
                issues.append(f"{col} out-of-range")
    issues = sorted(set(issues))
    if len(issues) == 0:
        return "None (potable)"
    return "; ".join(issues)

# ---------------------------
# Apply Logic
# ---------------------------
df["diseases"] = df.apply(assign_diseases, axis=1)
df["quality"] = df["diseases"].apply(lambda x: "good" if x == "None (potable)" else "bad")

# Save output
df.to_csv(output_path, index=False)
print("✅ Processing complete. Results saved to", output_path)
