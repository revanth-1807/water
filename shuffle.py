import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("water_potability_shuffled.csv")

# Shuffle dataset to avoid order bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Balance dataset (optional if unbalanced)
bad_data = df[df["quality"] == "bad"]
good_data = df[df["quality"] == "good"]
good_upsampled = good_data.sample(len(bad_data), replace=True, random_state=42)
df_balanced = pd.concat([bad_data, good_upsampled]).sample(frac=1, random_state=42)

# Inputs and outputs
X = df_balanced[['ph','Hardness','Solids','Chloramines','Sulfate',
                 'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']]
y_quality = df_balanced['quality']
y_diseases = df_balanced['diseases']

# -----------------------------
# 2. Encode Labels
# -----------------------------
le_quality = LabelEncoder()
le_diseases = LabelEncoder()
yq_enc = le_quality.fit_transform(y_quality)
yd_enc = le_diseases.fit_transform(y_diseases)

# Convert to one-hot
yq_onehot = tf.keras.utils.to_categorical(yq_enc)
yd_onehot = tf.keras.utils.to_categorical(yd_enc)

# -----------------------------
# 3. Scale Inputs
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, yq_train, yq_test, yd_train, yd_test = train_test_split(
    X_scaled, yq_onehot, yd_onehot, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Build Model
# -----------------------------
inputs = Input(shape=(X_train.shape[1],))
h = Dense(128, activation='relu')(inputs)
h = Dense(64, activation='relu')(h)

out_quality = Dense(yq_onehot.shape[1], activation='softmax', name='quality')(h)
out_diseases = Dense(yd_onehot.shape[1], activation='softmax', name='diseases')(h)

model = Model(inputs, [out_quality, out_diseases])

model.compile(
    optimizer='adam',
    loss={'quality': 'categorical_crossentropy', 'diseases': 'categorical_crossentropy'},
    metrics={'quality': 'accuracy', 'diseases': 'accuracy'}
)

# -----------------------------
# 5. Train Model
# -----------------------------
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)

model.fit(
    X_train, {'quality': yq_train, 'diseases': yd_train},
    validation_data=(X_test, {'quality': yq_test, 'diseases': yd_test}),
    epochs=50, batch_size=32, callbacks=[checkpoint]
)

# -----------------------------
# 6. Save Preprocessors
# -----------------------------
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_quality, open("le_quality.pkl", "wb"))
pickle.dump(le_diseases, open("le_diseases.pkl", "wb"))
