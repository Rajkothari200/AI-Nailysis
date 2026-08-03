"""
AI Nailysis V2 - Core Diagnostic Pipeline Engine
=================================================
Executes 3-Stage Diagnostic Pipeline:
1. Multi-Spectral Nail Polish & Artificial Fake Nail Detection
2. Stage-1 Binary Classification (Healthy vs Anomalous)
3. Stage-2 Multi-Class Pathological Classification
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

# ==============================
# CONFIG & PATHS
# ==============================

STAGE1_IMG_SIZE = 300
STAGE2_IMG_SIZE = 384
POLISH_IMG_SIZE = 224

base_dir = os.path.dirname(os.path.abspath(__file__))

stage1_weights_finetuned = os.path.join(base_dir, "Stage1", "stage1_finetuned_weights.h5")
stage1_weights_base = os.path.join(base_dir, "Stage1", "stage1_best_weights.h5")

stage2_weights_finetuned = os.path.join(base_dir, "Stage2", "stage2_finetuned_weights.h5")
stage2_weights_base = os.path.join(base_dir, "Stage2", "stage2_best_weights.h5")

polish_weights_path = os.path.join(base_dir, "Stage_Polish", "polish_detector_weights.h5")

classes = ['clubbing', 'cyanosis', 'melanoma', 'onychogryphosis', 'onychomycosis', 'psoriasis']

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

# ==============================
# MODEL BUILD & WEIGHT LOADING
# ==============================

# 1. POLISH MODEL
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
    print("[Warning] Polish detector weights not found.")

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

stage1_loaded = False
if os.path.exists(stage1_weights_base):
    try:
        model_stage1.load_weights(stage1_weights_base)
        print("[OK] Stage-1 baseline weights loaded.")
        stage1_loaded = True
    except Exception as e:
        print(f"[Warning] Error loading Stage-1 baseline weights: {e}")

if not stage1_loaded and os.path.exists(stage1_weights_finetuned):
    try:
        model_stage1.load_weights(stage1_weights_finetuned)
        print("[OK] Stage-1 fine-tuned weights loaded.")
    except Exception as e:
        print(f"[Error] Error loading Stage-1 fine-tuned weights: {e}")

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

stage2_loaded = False
if os.path.exists(stage2_weights_base):
    try:
        model_stage2.load_weights(stage2_weights_base)
        print("[OK] Stage-2 baseline weights loaded.")
        stage2_loaded = True
    except Exception as e:
        print(f"[Warning] Error loading Stage-2 baseline weights: {e}")

if not stage2_loaded and os.path.exists(stage2_weights_finetuned):
    try:
        model_stage2.load_weights(stage2_weights_finetuned)
        print("[OK] Stage-2 fine-tuned weights loaded.")
    except Exception as e:
        print(f"[Error] Error loading Stage-2 fine-tuned weights: {e}")


# ==============================
# PIPELINE FUNCTION INTERFACE
# ==============================

def detect_polish_and_fake_nails(img_rgb, base_prob):
    """
    Multi-Spectral Polish & Artificial Fake Nail Detector.
    Combines MobileNet polish model score with LAB/HSV color space heuristics.
    """
    score = float(base_prob)
    h, w = img_rgb.shape[:2]
    
    # Extract center 60% nail bed ROI
    crop = img_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    if crop.size > 0:
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        
        mean_l = np.mean(crop_lab[:, :, 0])
        mean_a = np.mean(crop_lab[:, :, 1])
        mean_b = np.mean(crop_lab[:, :, 2])
        
        mean_s = np.mean(crop_hsv[:, :, 1])
        mean_v = np.mean(crop_hsv[:, :, 2])
        
        std_l = np.std(crop_lab[:, :, 0])
        std_a = np.std(crop_lab[:, :, 1])
        std_b = np.std(crop_lab[:, :, 2])

        # A) Dark Matte Polish (Black, Dark Grey, Navy, Burgundy): L < 95 or V < 95
        is_dark_matte = (mean_l < 95 or mean_v < 95)
        
        # B) Non-skin artificial synthetic colors (Blue, Green, Black, Neon, Purple, Intense Red):
        # Natural skin/nail beds are pinkish (a* > 132). Non-pink artificial colors have a* < 129 or extreme saturation S > 95
        is_synthetic_color = (mean_a < 128 or (mean_s > 95 and mean_a < 133))
        
        # C) Ultra-smooth artificial acrylic / polish texture (low color standard deviation)
        is_synthetic_smooth = (std_l < 15.0 and std_a < 10.0 and std_b < 10.0)

        if is_dark_matte or is_synthetic_color:
            score = max(score, 0.96)
        elif is_synthetic_smooth and mean_s > 60:
            score = max(score, 0.88)

    detected = bool(score >= 0.5)
    confidence = float(score if detected else 1.0 - score)
    return detected, round(confidence * 100, 1)


def analyze_image_bgr(img_bgr):
    """
    Runs full 3-stage diagnostic pipeline on BGR image input.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. POLISH & ARTIFICIAL NAIL DETECTION
    img_polish = cv2.resize(img_rgb, (POLISH_IMG_SIZE, POLISH_IMG_SIZE))
    img_polish = np.expand_dims(img_polish, axis=0)
    img_polish = preprocess_mobilenet(img_polish)
    
    mobilenet_prob = model_polish.predict(img_polish, verbose=0)[0][0]
    polish_detected, polish_confidence = detect_polish_and_fake_nails(img_rgb, mobilenet_prob)
    
    # 2. STAGE 1 PREDICTION
    img1 = cv2.resize(img_rgb, (STAGE1_IMG_SIZE, STAGE1_IMG_SIZE))
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_efficientnet(img1)
    
    healthy_prob = model_stage1.predict(img1, verbose=0)[0][0]
    healthy = bool(healthy_prob >= 0.5)
    
    results = {
        "polish_detected": polish_detected,
        "polish_confidence": polish_confidence,
        "healthy": healthy,
        "disease_probability": round((1.0 - healthy_prob) * 100, 1)
    }
    
    # 3. STAGE 2 PREDICTION
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
