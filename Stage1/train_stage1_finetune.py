import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 15

train_dir = r"Stage1\train"
val_dir = r"Stage1\val"
test_dir = r"Stage1\test"

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
# MODEL REBUILD AND WEIGHTS LOADING
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

# Load original best weights
original_weights_path = r"Stage1\stage1_best_weights.h5"
if os.path.exists(original_weights_path):
    print(f"Loading weights from {original_weights_path}")
    model.load_weights(original_weights_path)
else:
    print(f"⚠️ Warning: original weights not found at {original_weights_path}, using ImageNet weights...")
    base_model_imagenet = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.set_weights(base_model_imagenet.get_weights())

# ==============================
# UNFREEZING FOR FINE-TUNING
# ==============================

print("\nUnfreezing top layers of base model for fine-tuning...")
base_model.trainable = True

# Freeze all layers except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
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
        filepath=r"Stage1\stage1_finetuned_weights.h5",
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

print("\nStarting Stage-1 Fine-Tuning...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nStage-1 Fine-tuning Complete.")

# ==============================
# EVALUATE
# ==============================

print("\nLoading best fine-tuned weights for evaluation...")
if os.path.exists(r"Stage1\stage1_finetuned_weights.h5"):
    model.load_weights(r"Stage1\stage1_finetuned_weights.h5")
else:
    print("Fine-tuned weights file not found, evaluating current weights.")

print("\nEvaluating on test set...")
results = model.evaluate(test_gen)

print("\nFine-tuned Test Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\n✅ Stage-1 Fine-Tuning script run finished.")
