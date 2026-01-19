import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===== CONFIG =====
DATA_DIR = "sequences"
EPOCHS = 25
BATCH_SIZE = 8
# ==================

X = []
y = []
labels = []

for i, file in enumerate(sorted(os.listdir(DATA_DIR))):
    if not file.endswith(".npy"):
        continue

    activity = file.replace(".npy", "")
    data = np.load(os.path.join(DATA_DIR, file))

    X.append(data)
    y.extend([i] * len(data))
    labels.append(activity)

X = np.vstack(X)
y = np.array(y)

y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ===== Evaluation =====
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_labels,
    labels=list(range(len(labels))),
    target_names=labels,
    zero_division=0
))



cm = confusion_matrix(
    y_true,
    y_pred_labels,
    labels=list(range(len(labels)))
)


plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.tight_layout()
plt.show()
