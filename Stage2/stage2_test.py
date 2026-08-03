import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 384

weights_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\stage2_best_weights.h5"

classes = [
    "clubbing",
    "cyanosis",
    "melanoma",
    "onychogryphosis",
    "onychomycosis",
    "psoriasis"
]

# ==============================
# BUILD MODEL
# ==============================

base_model = EfficientNetB4(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

model.load_weights(weights_path)

print("✅ Stage-2 model loaded")

# ==============================
# INPUT IMAGE
# ==============================

image_path = input("\nEnter image path: ")

img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==============================
# PREPROCESS
# ==============================

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# ==============================
# PREDICTION
# ==============================

pred = model.predict(img)

idx = np.argmax(pred)
confidence = pred[0][idx]

predicted_class = classes[idx]

# ==============================
# OUTPUT
# ==============================

print("\n🧠 Stage-2 Prediction:")
print("Disease:", predicted_class.capitalize())
print("Confidence:", round(float(confidence) * 100, 2), "%")

# Optional: show all class probabilities
print("\n🔍 All Class Probabilities:")
for i, cls in enumerate(classes):
    print(f"{cls}: {round(float(pred[0][i]) * 100, 2)}%")