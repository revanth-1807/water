import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# 0. Define Hyperparameters
# -----------------------------
HYPERPARAMS = {
    "epochs": 200,               # max epochs
    "batch_size": 64,            # batch size
    "lr0": 1e-3,                 # initial learning rate
    "lrf": 0.1,                  # final LR factor (lr = lr0 * lrf)
    "weight_decay": 1e-5,        # L2 regularization strength
    "hidden_units1": 256,
    "hidden_units2": 128,
    "dropout_rate": 0.3,
    "patience": 30,              # early stopping patience
    "optimizer": "adam"          # optimizer: adam, sgd, rmsprop
}

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("water_potability_shuffled.csv")

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
# 5. Learning Rate Scheduler
# -----------------------------
def lr_scheduler(epoch):
    return HYPERPARAMS["lr0"] * (HYPERPARAMS["lrf"] ** (epoch / HYPERPARAMS["epochs"]))

lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

# -----------------------------
# 6. Build Multi-output Neural Network
# -----------------------------
inputs = Input(shape=(X_train.shape[1],))

h = Dense(HYPERPARAMS["hidden_units1"], activation='relu',
          kernel_regularizer=regularizers.l2(HYPERPARAMS["weight_decay"]))(inputs)
h = Dropout(HYPERPARAMS["dropout_rate"])(h)
h = Dense(HYPERPARAMS["hidden_units2"], activation='relu',
          kernel_regularizer=regularizers.l2(HYPERPARAMS["weight_decay"]))(h)
h = Dropout(HYPERPARAMS["dropout_rate"])(h)

# Two output branches
out_quality = Dense(y_quality_onehot.shape[1], activation='softmax', name="quality")(h)
out_diseases = Dense(y_diseases_onehot.shape[1], activation='softmax', name="diseases")(h)

model = Model(inputs=inputs, outputs=[out_quality, out_diseases])

# -----------------------------
# 7. Optimizer Choice
# -----------------------------
if HYPERPARAMS["optimizer"].lower() == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS["lr0"])
elif HYPERPARAMS["optimizer"].lower() == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=HYPERPARAMS["lr0"], momentum=0.9)
elif HYPERPARAMS["optimizer"].lower() == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=HYPERPARAMS["lr0"])
else:
    raise ValueError("Unsupported optimizer in HYPERPARAMS['optimizer']")

# Compile
model.compile(
    optimizer=optimizer,
    loss={'quality': 'categorical_crossentropy', 'diseases': 'categorical_crossentropy'},
    metrics={'quality': ['accuracy'], 'diseases': ['accuracy']}
)

# -----------------------------
# 8. Save Best Model & Early Stopping
# -----------------------------
checkpoint = ModelCheckpoint("best_model3.keras", monitor="val_loss",
                             save_best_only=True, mode="min", verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=HYPERPARAMS["patience"],
                           restore_best_weights=True)

# -----------------------------
# 9. Train Model
# -----------------------------
history = model.fit(
    X_train, {"quality": yq_train, "diseases": yd_train},
    validation_data=(X_test, {"quality": yq_test, "diseases": yd_test}),
    epochs=HYPERPARAMS["epochs"],
    batch_size=HYPERPARAMS["batch_size"],
    callbacks=[checkpoint, early_stop, lr_callback],
    verbose=1
)

# -----------------------------
# 10. Evaluate Model
# -----------------------------
results = model.evaluate(X_test, {"quality": yq_test, "diseases": yd_test}, verbose=0)

print("\n📊 Model Evaluation:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# -----------------------------
# 11. Plot Training Graphs
# -----------------------------
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

# Loss plot
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# -----------------------------
# 12. Prediction Example
# -----------------------------
sample = np.array([[7.2, 200, 12000, 3.5, 250, 400, 10, 70, 4]])
sample_scaled = scaler.transform(sample)

pred_quality, pred_diseases = model.predict(sample_scaled)

predicted_quality = le_quality.inverse_transform([np.argmax(pred_quality.flatten())])[0]
predicted_disease = le_diseases.inverse_transform([np.argmax(pred_diseases.flatten())])[0]

if predicted_quality.lower() == "good":
    predicted_disease = "No Contamination"

print("\n🔮 Predicted Quality:", predicted_quality)
print("🔮 Predicted Disease:", predicted_disease)

# -----------------------------
# 13. Save Scaler & Label Encoders
# -----------------------------
pickle.dump(scaler, open("scaler1.pkl", "wb"))
pickle.dump(le_quality, open("le_quality1.pkl", "wb"))
pickle.dump(le_diseases, open("le_diseases1.pkl", "wb"))