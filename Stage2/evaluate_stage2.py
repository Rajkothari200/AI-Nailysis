import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 384
BATCH_SIZE = 8

test_dir = r"Stage2\test"
weights_path = r"Stage2\stage2_best_weights.h5"

# ==============================
# DATA GENERATOR
# ==============================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\nClass indices:", test_gen.class_indices)

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

print("\n[OK] Stage-2 model loaded")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# EVALUATE
# ==============================

results = model.evaluate(test_gen)

print("\nTest Loss:", results[0])
print("Test Accuracy:", results[1])

# ==============================
# PREDICTIONS
# ==============================

predictions = model.predict(test_gen)

y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

class_names = list(test_gen.class_indices.keys())

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:\n")
print(report)