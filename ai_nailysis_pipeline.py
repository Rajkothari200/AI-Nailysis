"""
AI Nailysis V2 - Core Diagnostic Pipeline Engine
=================================================
Executes 3-Stage Diagnostic Pipeline:
1. Stage-1 Binary Classification (Healthy vs Anomalous)
2. Stage-2 Multi-Class Pathological Classification
3. Precision Nail Polish & Artificial Fake Nail Detection
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

def precision_detect_polish(img_rgb, mobilenet_prob, disease_detected, disease_confidence):
    """
    Precision Polish & Artificial Fake Nail Detector.
    Pathology Priority Rule: If a disease signature is present (disease_detected is True / healthy is False),
    clinical disease diagnosis takes priority. Polish is marked False unless mobilenet_prob >= 0.85.
    
    For Healthy Nails (disease_detected is False):
    Detects solid polish, manicures, and nail art using MobileNetV2 + HSV chromaticity/contrast analysis.
    """
    mobilenet_score = float(mobilenet_prob)
    
    # 1. PATHOLOGY PRIORITY RULE: Diseased nails default to False for polish
    if disease_detected:
        if mobilenet_score < 0.85:
            return False, round((1.0 - mobilenet_score) * 100, 1)
        else:
            return True, round(mobilenet_score * 100, 1)

    # 2. HEALTHY NAIL COSMETIC DETECTION (Polish / Art / Manicures)
    score = mobilenet_score

    # Compute HSV color metrics for nail art & vibrant polish on healthy nails
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # High saturation ratio (vibrant red/burgundy/pink/blue polish)
    high_sat_ratio = np.mean(s_chan > 100)
    # High standard deviation in saturation & value (nail art patterns like checkered/hearts/french tip)
    s_std = np.std(s_chan)
    v_std = np.std(v_chan)

    if high_sat_ratio > 0.08 or s_std > 45 or v_std > 50 or mobilenet_score >= 0.50:
        score = max(score, 0.78 if (high_sat_ratio > 0.08 or s_std > 45 or v_std > 50) else score)

    detected = bool(score >= 0.50)
    confidence = float(score if detected else 1.0 - score)
    return detected, round(confidence * 100, 1)


def analyze_image_bgr(img_bgr, norm_img=None, full_img_bgr=None):
    """
    Runs full 3-stage diagnostic pipeline on BGR image input.
    Optionally accepts norm_img for illumination-normalized disease classification
    and full_img_bgr for full-frame analysis.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Use norm_img for Stage 1/2 disease classification if provided
    if norm_img is not None:
        disease_img_rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
    else:
        disease_img_rgb = img_rgb

    # 1. STAGE 1 PREDICTION (HEALTHY VS ANOMALOUS)
    img1 = cv2.resize(disease_img_rgb, (STAGE1_IMG_SIZE, STAGE1_IMG_SIZE))
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_efficientnet(img1)
    
    healthy_prob = model_stage1.predict(img1, verbose=0)[0][0]
    healthy = bool(healthy_prob >= 0.5)
    disease_prob_pct = round((1.0 - healthy_prob) * 100, 1)
    
    # 2. STAGE 2 PREDICTION (DISEASE CLASSIFICATION IF ANOMALOUS)
    disease_name = "healthy"
    disease_conf = round(healthy_prob * 100, 1)
    info_dict = None
    all_conf = None
    
    if not healthy:
        img2 = cv2.resize(disease_img_rgb, (STAGE2_IMG_SIZE, STAGE2_IMG_SIZE))
        img2 = np.expand_dims(img2, axis=0)
        img2 = preprocess_efficientnet(img2)
        
        pred = model_stage2.predict(img2, verbose=0)[0]
        idx = np.argmax(pred)
        disease_name = classes[idx]
        disease_conf = round(float(pred[idx]) * 100, 1)
        info_dict = disease_info[disease_name]
        
        all_conf = {}
        for c, score in zip(classes, pred):
            all_conf[c] = float(score) * 100

    # 3. POLISH & ARTIFICIAL NAIL DETECTION
    img_polish = cv2.resize(img_rgb, (POLISH_IMG_SIZE, POLISH_IMG_SIZE))
    img_polish = np.expand_dims(img_polish, axis=0)
    img_polish = preprocess_mobilenet(img_polish)
    
    mobilenet_prob = model_polish.predict(img_polish, verbose=0)[0][0]
    
    is_diseased = (not healthy)
    polish_detected, polish_confidence = precision_detect_polish(
        img_rgb, mobilenet_prob, disease_detected=is_diseased, disease_confidence=disease_conf
    )
    
    results = {
        "polish_detected": polish_detected,
        "polish_confidence": polish_confidence,
        "healthy": healthy,
        "disease_probability": disease_prob_pct,
        "disease": disease_name,
        "disease_confidence": disease_conf,
        "info": info_dict,
        "all_confidences": all_conf
    }
    
    return results
