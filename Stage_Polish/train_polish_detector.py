import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
dataset_root = "Stage_Polish"

train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")
test_dir = os.path.join(dataset_root, "test")

# If --eval is passed, we only run evaluation
is_eval = len(sys.argv) > 1 and sys.argv[1] == "--eval"

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", train_gen.class_indices)

# ==============================
# BUILD MODEL
# ==============================

base_model = MobileNetV2(
    weights='imagenet' if not is_eval else None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

weights_path = os.path.join(dataset_root, "polish_detector_weights.h5")

if is_eval:
    print(f"Loading weights from {weights_path} for evaluation...")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print("Weights file not found!")
        sys.exit(1)
        
    results = model.evaluate(test_gen)
    print("\nEvaluation Results on Test Set:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
    sys.exit(0)

# ==============================
# CALLBACKS & TRAINING
# ==============================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
]

print("Starting training of polish detector...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Load best weights and evaluate
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    
print("\nEvaluating on test set...")
results = model.evaluate(test_gen)
print("\nFinal Test Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("[OK] Polish detector training finished.")
