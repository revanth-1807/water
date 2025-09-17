import pandas as pd
import numpy as np

# Input / Output paths
input_path = "water_potability_with_labels.csv"
output_path = "water_potability_with_labels1.csv"

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(input_path)

# Get numeric columns (excluding target if present)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Potability" in numeric_cols:
    numeric_cols.remove("Potability")

# Fill missing numeric values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# 2. Quality Column Assignment
# -----------------------------
# If Potability exists: use it directly
if "Potability" in df.columns:
    df["quality"] = df["Potability"].apply(lambda x: "good" if x == 1 else "bad")
else:
    # If no Potability column, decide using 5-95 percentile thresholds
    lo = df[numeric_cols].quantile(0.05)
    hi = df[numeric_cols].quantile(0.95)

    def decide_quality(row):
        for col in numeric_cols:
            if row[col] < lo[col] or row[col] > hi[col]:
                return "bad"
        return "good"

    df["quality"] = df.apply(decide_quality, axis=1)

# -----------------------------
# 3. Disease Mapping
# -----------------------------
disease_map = {
    "ph": "Skin/eye irritation; Gastrointestinal upset",
    "Hardness": "Taste/mineral effects; kidney stone risk",
    "Solids": "Gastrointestinal illness (diarrhea, stomach upset)",
    "Chloramines": "Bacterial risk; irritation",
    "Sulfate": "Laxative effects / diarrhea at high concentrations",
    "Conductivity": "Dissolved solids → GI issues",
    "Organic_carbon": "Disinfection byproducts risk",
    "Trihalomethanes": "Cancer risk; irritation",
    "Turbidity": "Waterborne pathogens risk"
}
available_map = {col: disease_map[col] for col in disease_map if col in df.columns}

# Reference ranges for "good" water
good_df = df[df["quality"] == "good"]
lo_ref = good_df[numeric_cols].quantile(0.05)
hi_ref = good_df[numeric_cols].quantile(0.95)

# -----------------------------
# 4. Assign Diseases
# -----------------------------
def assign_diseases(row):
    if row["quality"] == "good":
        return "None (potable)"  # ✅ Good water → No disease

    issues = []
    for col in numeric_cols:
        val = row[col]
        if val < lo_ref[col] or val > hi_ref[col]:
            issues.append(available_map.get(col, f"{col} out-of-range"))

    return "; ".join(sorted(set(issues))) if issues else "Unknown contamination"

df["diseases"] = df.apply(assign_diseases, axis=1)

# -----------------------------
# 5. Save Clean Dataset
# -----------------------------
df.to_csv(output_path, index=False)
print(f"✅ Clean dataset saved to: {output_path}")
