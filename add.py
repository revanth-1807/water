import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("your_dataset.csv")

# Features (Inputs)
X = df[['A', 'B', 'C', 'D']]

# Targets (Outputs - strings)
y = df[['E', 'F']]

# -----------------------------
# 2. Encode String Labels
# -----------------------------
le_E = LabelEncoder()
le_F = LabelEncoder()

y_encoded = pd.DataFrame({
    'E': le_E.fit_transform(y['E']),
    'F': le_F.fit_transform(y['F'])
})

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Model Training
# -----------------------------
base_model = RandomForestClassifier(random_state=42)
model = MultiOutputClassifier(base_model)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("✅ Model Evaluation")
print("Accuracy for E:", accuracy_score(y_test['E'], y_pred[:, 0]))
print("Accuracy for F:", accuracy_score(y_test['F'], y_pred[:, 1]))
print("-" * 40)

# -----------------------------
# 6. Prediction Function
# -----------------------------
def predict_labels(a, b, c, d):
    new_data = [[a, b, c, d]]
    pred_encoded = model.predict(new_data)
    pred_E = le_E.inverse_transform([pred_encoded[0][0]])[0]
    pred_F = le_F.inverse_transform([pred_encoded[0][1]])[0]
    return pred_E, pred_F

# -----------------------------
# 7. Test with Manual Input
# -----------------------------
try:
    print("🔮 Enter values for A, B, C, D to predict E and F:")
    a = int(input("Enter A: "))
    b = int(input("Enter B: "))
    c = int(input("Enter C: "))
    d = int(input("Enter D: "))

    pred_E, pred_F = predict_labels(a, b, c, d)
    print("\n🎯 Predicted E:", pred_E)
    print("🎯 Predicted F:", pred_F)

except Exception as e:
    print("⚠️ Error:", e)
