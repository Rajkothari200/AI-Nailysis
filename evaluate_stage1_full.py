import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 300
BATCH_SIZE = 8

test_dir = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage1\test"
weights_path = r"D:\AI Nail Analysis\AI_Nailysis_stage1\stage1_best_weights.h5"

# -----------------------
# DATA LOADER
# -----------------------

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", test_gen.class_indices)

# -----------------------
# MODEL
# -----------------------

base_model = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.load_weights(weights_path)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Stage-1 model loaded")

# -----------------------
# EVALUATION
# -----------------------

results = model.evaluate(test_gen)

print("\nTest Loss:", results[0])
print("Test Accuracy:", results[1])

# -----------------------
# PREDICTIONS
# -----------------------

pred = model.predict(test_gen)

y_pred = (pred > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))