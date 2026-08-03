import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 25

train_dir = r"D:\AI Nail Analysis\AI_Nailysis_stage1\train"
val_dir = r"D:\AI Nail Analysis\AI_Nailysis_stage1\val"
test_dir = r"D:\AI Nail Analysis\AI_Nailysis_stage1\test"

print("TensorFlow Version:", tf.__version__)

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
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
# MODEL
# ==============================

base_model = EfficientNetB3(
    weights='imagenet',
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
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# ==============================
# CALLBACKS (SAFE VERSION)
# ==============================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=r"D:\AI Nail Analysis\AI_Nailysis_stage1\stage1_best_weights.h5",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,   # <-- CRITICAL
        mode='min',
        verbose=1
    )
]

# ==============================
# TRAIN
# ==============================

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete.")

# ==============================
# LOAD BEST WEIGHTS
# ==============================

model.load_weights(r"D:\AI Nail Analysis\AI_Nailysis_stage1\stage1_best_weights.h5")

print("\nEvaluating on test set...")
results = model.evaluate(test_gen)

print("\nTest Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\n✅ Stage-1 complete.")