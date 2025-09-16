import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("water_potability_with_cleaned_labels.csv")

# Inputs (numeric features)
X = df[['ph','Hardness','Solids','Chloramines','Sulfate',
        'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']]

# Outputs (categorical targets)
y_quality = df['quality']
y_diseases = df['diseases']

# -----------------------------
# 2. Encode String Labels
# -----------------------------
le_quality = LabelEncoder()
le_diseases = LabelEncoder()

y_quality_enc = le_quality.fit_transform(y_quality)
y_diseases_enc = le_diseases.fit_transform(y_diseases)

# Convert to one-hot
y_quality_onehot = tf.keras.utils.to_categorical(y_quality_enc)
y_diseases_onehot = tf.keras.utils.to_categorical(y_diseases_enc)

# -----------------------------
# 3. Scale Inputs
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, yq_train, yq_test, yd_train, yd_test = train_test_split(
    X_scaled, y_quality_onehot, y_diseases_onehot, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Build Multi-output Neural Network
# -----------------------------
inputs = Input(shape=(X_train.shape[1],))

# Shared hidden layers
h = Dense(128, activation='relu')(inputs)
h = Dense(64, activation='relu')(h)

# Two output branches
out_quality = Dense(y_quality_onehot.shape[1], activation='softmax', name="quality")(h)
out_diseases = Dense(y_diseases_onehot.shape[1], activation='softmax', name="diseases")(h)

# Define model
model = Model(inputs=inputs, outputs=[out_quality, out_diseases])

# Compile (✅ FIXED metrics per output)
model.compile(
    optimizer='adam',
    loss={
        'quality': 'categorical_crossentropy',
        'diseases': 'categorical_crossentropy'
    },
    metrics={
        'quality': ['accuracy'],
        'diseases': ['accuracy']
    }
)

# -----------------------------
# 6. Save Best Model Callback
# -----------------------------
checkpoint = ModelCheckpoint(
    "best_model.keras",   # filename (Keras format)
    monitor="val_loss",   # monitor validation loss
    save_best_only=True,  # only save the best one
    mode="min",
    verbose=1
)

# -----------------------------
# 7. Train Model
# -----------------------------
history = model.fit(
    X_train, {"quality": yq_train, "diseases": yd_train},
    validation_data=(X_test, {"quality": yq_test, "diseases": yd_test}),
    epochs=50, batch_size=32,
    callbacks=[checkpoint]
)

# -----------------------------
# 8. Evaluate Model on Test Data
# -----------------------------
results = model.evaluate(X_test, {"quality": yq_test, "diseases": yd_test}, verbose=0)

print("\n📊 Model Evaluation:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# -----------------------------
# 9. Plot Training Graphs
# -----------------------------
# Accuracy
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['quality_accuracy'], label='Train Quality Acc')
plt.plot(history.history['val_quality_accuracy'], label='Val Quality Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Quality Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['diseases_accuracy'], label='Train Diseases Acc')
plt.plot(history.history['val_diseases_accuracy'], label='Val Diseases Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Diseases Accuracy')
plt.legend()

plt.show()

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# -----------------------------
# 10. Prediction Example
# -----------------------------
sample = np.array([[7.2, 200, 12000, 3.5, 250, 400, 10, 70, 4]])  # Example values
sample_scaled = scaler.transform(sample)

pred_quality, pred_diseases = model.predict(sample_scaled)

print("\n🔮 Predicted quality:", le_quality.inverse_transform([np.argmax(pred_quality)]))
print("🔮 Predicted disease:", le_diseases.inverse_transform([np.argmax(pred_diseases)]))


import pickle

# Save the trained scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save the label encoders
pickle.dump(le_quality, open("le_quality.pkl", "wb"))
pickle.dump(le_diseases, open("le_diseases.pkl", "wb"))

# Save the model (optional, already saved by checkpoint)
model.save("best_model.keras")
