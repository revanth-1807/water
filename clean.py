import pandas as pd
import numpy as np

# Load your dataset again
input_path = "water_potability_with_labels.csv"
df = pd.read_csv(input_path)

# Impute missing numeric values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Potability" in numeric_cols:
    numeric_cols.remove("Potability")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Create 'quality' column based on Potability
df["quality"] = df["Potability"].apply(lambda x: "good" if x == 1 else "bad")

# Compute potable ranges (5th–95th percentile)
potable_df = df[df["Potability"] == 1]
potable_lo = potable_df[numeric_cols].quantile(0.05)
potable_hi = potable_df[numeric_cols].quantile(0.95)

# Map of features → health issues
disease_map = {
    "ph": "Skin/eye irritation; Gastrointestinal upset",
    "Hardness": "Taste/mineral effects; kidney stone risk",
    "Solids": "Gastrointestinal illness",
    "Chloramines": "Bacterial risk; irritation",
    "Sulfate": "Laxative effects / diarrhea",
    "Conductivity": "Dissolved solids → GI issues",
    "Organic_carbon": "Disinfection byproducts risk",
    "Trihalomethanes": "Cancer risk; irritation",
    "Turbidity": "Waterborne pathogens risk"
}

# Function to assign diseases, keeping only top 2 issues
def assign_diseases(row):
    issues = []
    for col in numeric_cols:
        if col not in potable_lo or col not in potable_hi:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if val < potable_lo[col] or val > potable_hi[col]:
            if col in disease_map:
                issues.append(disease_map[col])
            else:
                issues.append(f"{col} out-of-range")
    
    # Deduplicate and keep only top 2
    issues = sorted(set(issues))[:2]
    
    if len(issues) == 0:
        return "None (potable)" if row["quality"] == "good" else "Unknown contamination"
    
    return "; ".join(issues)

# Apply function
df["diseases"] = df.apply(assign_diseases, axis=1)

# Save the cleaned-up dataset
output_path = "water_potability_with_cleaned_labels.csv"
df.to_csv(output_path, index=False)

print(f"Saved cleaned dataset to: {output_path}")
