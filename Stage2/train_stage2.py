import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# CONFIG
# ==============================

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 25

train_dir = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\train"
val_dir   = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\val"
test_dir  = r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\test"

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
    brightness_range=[0.8,1.2]
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
# MODEL
# ==============================

base_model = EfficientNetB4(
    weights='imagenet',
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
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
        filepath=r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\stage2_best_weights.h5",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  
        mode='min',
        verbose=1
    )
]

# ==============================
# TRAIN (PHASE 1)
# ==============================

print("\nStarting Stage-2 Training (Phase 1)...")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# LOAD BEST WEIGHTS
# ==============================

model.load_weights(
r"D:\AI Nail Analysis\AI_Nailysis_stage1\Stage2\stage2_best_weights.h5"
)

print("\nEvaluating on test set...")

results = model.evaluate(test_gen)

print("\nTest Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\n✅ Stage-2 training complete.")