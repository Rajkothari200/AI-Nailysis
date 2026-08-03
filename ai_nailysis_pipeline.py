import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

# ==============================
# CONFIG
# ==============================

STAGE1_IMG_SIZE = 300
STAGE2_IMG_SIZE = 384
POLISH_IMG_SIZE = 224

# Relative paths for weights
stage1_weights_finetuned = r"Stage1\stage1_finetuned_weights.h5"
stage1_weights_base = r"Stage1\stage1_best_weights.h5"

stage2_weights_finetuned = r"Stage2\stage2_finetuned_weights.h5"
stage2_weights_base = r"Stage2\stage2_best_weights.h5"

polish_weights_path = r"Stage_Polish\polish_detector_weights.h5"

disease_info = {
    "clubbing": {
        "description": "Enlargement and rounding of fingertips often linked to chronic lung or heart disease.",
        "prevention": "Maintain lung health, avoid smoking, treat respiratory illnesses early.",
        "treatment": "Consult a doctor to diagnose underlying causes such as lung disease, heart disease, or gastrointestinal disorders."
    },
    "cyanosis": {
        "description": "Bluish discoloration of the nails caused by low oxygen levels in blood.",
        "prevention": "Maintain cardiovascular health, avoid smoking, manage lung conditions.",
        "treatment": "Seek medical evaluation immediately as it may indicate respiratory or cardiac issues."
    },
    "melanoma": {
        "description": "A dangerous form of skin cancer that may appear as dark streaks under the nail.",
        "prevention": "Protect skin from excessive UV exposure, monitor nail pigmentation changes.",
        "treatment": "Immediate dermatology consultation is required. Early diagnosis significantly improves survival."
    },
    "onychogryphosis": {
        "description": "Thickened and curved nails often caused by trauma or poor foot care.",
        "prevention": "Maintain proper nail hygiene, wear comfortable footwear.",
        "treatment": "Regular nail trimming, podiatrist consultation, sometimes surgical correction."
    },
    "onychomycosis": {
        "description": "Fungal infection of the nail causing discoloration and thickening.",
        "prevention": "Keep nails dry and clean, avoid sharing nail tools, wear breathable footwear.",
        "treatment": "Antifungal medications (topical or oral) prescribed by a healthcare professional."
    },
    "psoriasis": {
        "description": "Autoimmune condition causing nail pitting, discoloration, and thickening.",
        "prevention": "Manage stress, maintain skin care routines, follow dermatologist advice.",
        "treatment": "Topical steroids, vitamin D analogs, or systemic treatments prescribed by a dermatologist."
    }
}

classes = [
    "clubbing",
    "cyanosis",
    "melanoma",
    "onychogryphosis",
    "onychomycosis",
    "psoriasis"
]

# ==============================
# BUILD AND LOAD MODELS
# ==============================

# 1. POLISH DETECTOR MODEL
print("Loading Polish Detector model...")
base_model_polish = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(POLISH_IMG_SIZE, POLISH_IMG_SIZE, 3)
)
base_model_polish.trainable = False
model_polish = models.Sequential([
    base_model_polish,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

if os.path.exists(polish_weights_path):
    try:
        model_polish.load_weights(polish_weights_path)
        print("[OK] Polish detector model weights loaded.")
    except Exception as e:
        print(f"[Error] Error loading Polish detector weights: {e}")
else:
    print("[Warning] Polish detector weights not found. Model will output uncalibrated predictions.")

# 2. STAGE 1 MODEL (HEALTHY VS DISEASE)
print("Loading Stage-1 model...")
base_model1 = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(STAGE1_IMG_SIZE, STAGE1_IMG_SIZE, 3)
)
base_model1.trainable = False
model_stage1 = models.Sequential([
    base_model1,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Load fine-tuned weights if available, otherwise baseline weights
stage1_loaded = False
if os.path.exists(stage1_weights_finetuned):
    try:
        model_stage1.load_weights(stage1_weights_finetuned)
        print("[OK] Stage-1 fine-tuned weights loaded.")
        stage1_loaded = True
    except Exception as e:
        print(f"[Warning] Error loading Stage-1 fine-tuned weights: {e}. Falling back to baseline...")

if not stage1_loaded:
    if os.path.exists(stage1_weights_base):
        try:
            model_stage1.load_weights(stage1_weights_base)
            print("[OK] Stage-1 baseline weights loaded.")
        except Exception as e:
            print(f"[Error] Error loading Stage-1 baseline weights: {e}")
    else:
        print("[Warning] Stage-1 weights not found.")

# 3. STAGE 2 MODEL (DISEASE CLASSIFIER)
print("Loading Stage-2 model...")
base_model2 = EfficientNetB4(
    weights=None,
    include_top=False,
    input_shape=(STAGE2_IMG_SIZE, STAGE2_IMG_SIZE, 3)
)
base_model2.trainable = False
model_stage2 = models.Sequential([
    base_model2,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

# Load fine-tuned weights if available, otherwise baseline weights
stage2_loaded = False
if os.path.exists(stage2_weights_finetuned):
    try:
        model_stage2.load_weights(stage2_weights_finetuned)
        print("[OK] Stage-2 fine-tuned weights loaded.")
        stage2_loaded = True
    except Exception as e:
        print(f"[Warning] Error loading Stage-2 fine-tuned weights: {e}. Falling back to baseline...")

if not stage2_loaded:
    if os.path.exists(stage2_weights_base):
        try:
            model_stage2.load_weights(stage2_weights_base)
            print("[OK] Stage-2 baseline weights loaded.")
        except Exception as e:
            print(f"[Error] Error loading Stage-2 baseline weights: {e}")
    else:
        print("[Warning] Stage-2 weights not found.")


# ==============================
# PIPELINE FUNCTION INTERFACE
# ==============================

def analyze_image_bgr(img_bgr):
    """
    Takes a CV2 image (BGR format) and runs the entire AI Nailysis pipeline:
    1. Polish Detection
    2. Stage-1 Binary classification (Healthy vs Disease)
    3. Stage-2 Multi-class classification (if anomalous)
    Returns a dictionary of findings.
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. POLISH DETECTION
    img_polish = cv2.resize(img_rgb, (POLISH_IMG_SIZE, POLISH_IMG_SIZE))
    img_polish = np.expand_dims(img_polish, axis=0)
    img_polish = preprocess_mobilenet(img_polish)
    
    polish_prob = model_polish.predict(img_polish, verbose=0)[0][0]
    polish_detected = bool(polish_prob >= 0.5)
    polish_confidence = float(polish_prob if polish_detected else 1.0 - polish_prob)
    
    # 2. STAGE 1 PREDICTION
    img1 = cv2.resize(img_rgb, (STAGE1_IMG_SIZE, STAGE1_IMG_SIZE))
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_efficientnet(img1)
    
    # Sigmoid output represents the probability of index 1 (healthy)
    healthy_prob = model_stage1.predict(img1, verbose=0)[0][0]
    healthy = bool(healthy_prob >= 0.5)
    
    # Prepare results structure
    results = {
        "polish_detected": polish_detected,
        "polish_confidence": round(polish_confidence * 100, 1),
        "healthy": healthy,
        "disease_probability": round((1.0 - healthy_prob) * 100, 1)
    }
    
    # 3. STAGE 2 PREDICTION (ONLY RUN IF DISEASED)
    if not healthy:
        img2 = cv2.resize(img_rgb, (STAGE2_IMG_SIZE, STAGE2_IMG_SIZE))
        img2 = np.expand_dims(img2, axis=0)
        img2 = preprocess_efficientnet(img2)
        
        pred = model_stage2.predict(img2, verbose=0)[0]
        idx = np.argmax(pred)
        disease = classes[idx]
        confidence = pred[idx]
        
        results["disease"] = disease
        results["disease_confidence"] = round(float(confidence) * 100, 1)
        results["info"] = disease_info[disease]
        
        # Compile all confidences for charts
        all_conf = {}
        for c, score in zip(classes, pred):
            all_conf[c] = float(score) * 100
        results["all_confidences"] = all_conf
    else:
        results["disease"] = "healthy"
        results["disease_confidence"] = round(healthy_prob * 100, 1)
        results["info"] = None
        results["all_confidences"] = None
        
    return results


# ==============================
# CLI EXECUTION (LEGACY SUPPORT)
# ==============================

if __name__ == "__main__":
    print("\n--- AI Nailysis Pipeline (CLI Mode) ---")
    image_path = input("\nEnter image path: ")
    img = cv2.imread(image_path)
    
    if img is None:
        print("Image not found")
        exit()
        
    res = analyze_image_bgr(img)
    
    print("\n--- Diagnostic Report ---")
    print(f"Nail Polish Detected: {res['polish_detected']} ({res['polish_confidence']}%)")
    print(f"Result: {'HEALTHY NAIL' if res['healthy'] else 'DISEASE DETECTED'}")
    print(f"Disease Probability: {res['disease_probability']}%")
    
    if not res["healthy"]:
        print(f"\nDisease Identified: {res['disease'].capitalize()}")
        print(f"Confidence: {res['disease_confidence']}%")
        print(f"\nDescription:\n{res['info']['description']}")
        print(f"\nPrevention:\n{res['info']['prevention']}")
        print(f"\nPossible Treatment:\n{res['info']['treatment']}")
