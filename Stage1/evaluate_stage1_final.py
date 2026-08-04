import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 300
BATCH_SIZE = 16

test_dir = r"Stage1\test"
weights_path = r"Stage1\stage1_best_weights.h5"

print("TensorFlow Version:", tf.__version__)

# ==============================
# TEST GENERATOR
# ==============================

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", test_gen.class_indices)

# ==============================
# REBUILD MODEL ARCHITECTURE
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# LOAD TRAINED WEIGHTS
# ==============================

model.load_weights(weights_path)

print("\nWeights loaded successfully.")

# ==============================
# PREDICTIONS
# ==============================

pred_probs = model.predict(test_gen, verbose=0)

# ==============================
# THRESHOLD SWEEP
# ==============================

print("\n=== Threshold Evaluation ===")

for threshold in [0.5, 0.45, 0.4, 0.35, 0.3]:

    preds = (pred_probs > threshold).astype(int)
    cm = confusion_matrix(test_gen.classes, preds)

    # Since class mapping is {'disease': 0, 'healthy': 1}
    disease_recall = cm[0][0] / (cm[0][0] + cm[0][1])
    healthy_recall = cm[1][1] / (cm[1][0] + cm[1][1])
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)

    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Disease Recall (Sensitivity): {disease_recall:.4f}")
    print(f"Healthy Recall (Specificity): {healthy_recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# ==============================
# DEFAULT CLASSIFICATION REPORT (0.5)
# ==============================

print("\n=== Detailed Classification Report (Threshold=0.5) ===")
preds_default = (pred_probs > 0.5).astype(int)
print(classification_report(test_gen.classes, preds_default, target_names=['disease','healthy']))

print("\n[OK] Evaluation Complete.")