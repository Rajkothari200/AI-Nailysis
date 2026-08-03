import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 15

train_dir = r"Stage2\train"
val_dir   = r"Stage2\val"
test_dir  = r"Stage2\test"

print("TensorFlow Version:", tf.__version__)

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:", train_gen.class_indices)

# ==============================
# MODEL REBUILD AND WEIGHTS LOADING
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

# Load original best weights
original_weights_path = r"Stage2\stage2_best_weights.h5"
if os.path.exists(original_weights_path):
    print(f"Loading weights from {original_weights_path}")
    model.load_weights(original_weights_path)
else:
    print(f"⚠️ Warning: original weights not found at {original_weights_path}, using ImageNet weights...")
    base_model_imagenet = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.set_weights(base_model_imagenet.get_weights())

# ==============================
# UNFREEZING FOR FINE-TUNING
# ==============================

print("\nUnfreezing top layers of base model for fine-tuning...")
base_model.trainable = True

# Freeze all layers except the last 40 layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=r"Stage2\stage2_finetuned_weights.h5",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
]

# ==============================
# FINE-TUNE TRAINING
# ==============================

print("\nStarting Stage-2 Fine-Tuning...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nStage-2 Fine-tuning Complete.")

# ==============================
# EVALUATE
# ==============================

print("\nLoading best fine-tuned weights for evaluation...")
if os.path.exists(r"Stage2\stage2_finetuned_weights.h5"):
    model.load_weights(r"Stage2\stage2_finetuned_weights.h5")
else:
    print("Fine-tuned weights file not found, evaluating current weights.")

print("\nEvaluating on test set...")
results = model.evaluate(test_gen)

print("\nFine-tuned Test Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\n✅ Stage-2 Fine-Tuning script run finished.")
