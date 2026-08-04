import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

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


# BUILD MODEL


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


# LOAD IMAGE

image_path = input("Enter image path: ")

img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

input_img = np.expand_dims(img_resized, axis=0)
input_img = preprocess_input(input_img)


# PREDICTION


pred = model.predict(input_img, verbose=0)

class_idx = np.argmax(pred)
confidence = pred[0][class_idx]

print("\nPrediction:", classes[class_idx])
print("Confidence:", round(float(confidence)*100,2), "%")

# OCCLUSION MAP


patch_size = 40
stride = 20

heatmap = np.zeros((IMG_SIZE, IMG_SIZE))

baseline = confidence

for y in range(0, IMG_SIZE - patch_size, stride):
    for x in range(0, IMG_SIZE - patch_size, stride):

        occluded = img_resized.copy()

        occluded[y:y+patch_size, x:x+patch_size] = 128

        test_img = np.expand_dims(occluded, axis=0)
        test_img = preprocess_input(test_img)

        pred = model.predict(test_img, verbose=0)

        drop = baseline - pred[0][class_idx]

        heatmap[y:y+patch_size, x:x+patch_size] += drop

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# DISPLAY RESULTS


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Occlusion Heatmap")
plt.imshow(heatmap_color)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Important Regions")
plt.imshow(overlay)
plt.axis("off")

plt.show()