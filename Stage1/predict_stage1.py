import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 300

weights_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\stage1_best_weights.h5"

image_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\images.jpeg"
# replace this with your test image

# ==============================
# BUILD MODEL
# ==============================

base_model = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.load_weights(weights_path)

print("✅ Model loaded successfully")

# ==============================
# LOAD IMAGE
# ==============================

img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found. Check path.")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img = np.array(img)
img = np.expand_dims(img, axis=0)

img = preprocess_input(img)

# ==============================
# PREDICTION
# ==============================

prob = model.predict(img)[0][0]

print("\nPrediction probability:", prob)

if prob < 0.5:
    print("\n🔴 Result: DISEASED NAIL")
else:
    print("\n🟢 Result: HEALTHY NAIL")