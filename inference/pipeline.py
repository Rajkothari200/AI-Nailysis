"""
AI Nailysis V2 - Research-Grade Unified Inference Pipeline
============================================================
Sequentially executes: Image Quality Assessment -> Automatic Nail Segmentation ->
Color Normalization -> Multi-Task Neural Prediction -> Monte Carlo Dropout Confidence ->
GradCAM Explainability Overlay Generation.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.iqa import ImageQualityAssessor
from models.segmentation import NailSegmentationPipeline
from utils.color_norm import ColorNormalizer
from models.multi_task import MultiTaskAINailysisModel
from utils.xai import GradCAMExplainer
from utils.confidence import ConfidenceEstimator
from utils.logger import get_logger

logger = get_logger("InferencePipeline")


class AINailysisV2Pipeline:
    """
    End-to-End Clinical Vision Diagnostic Pipeline.
    """
    def __init__(self, config: Dict[str, Any], model_weights_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("system", {}).get("device") == "cuda" else "cpu")
        
        # Instantiate Sub-modules
        self.iqa_engine = ImageQualityAssessor(config)
        self.segmenter = NailSegmentationPipeline(config)
        self.normalizer = ColorNormalizer(config)
        self.confidence_estimator = ConfidenceEstimator(config)
        
        # Instantiate Multi-Task PyTorch Model
        self.model = MultiTaskAINailysisModel(config).to(self.device)
        self.model.eval()
        
        # Load PyTorch Weights if available
        if model_weights_path and os.path.exists(model_weights_path):
            try:
                state_dict = torch.load(model_weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded PyTorch V2 model weights from {model_weights_path}")
            except Exception as e:
                logger.warning(f"Could not load V2 model weights: {e}. Model operating in research baseline state.")

        # XAI Explainer
        self.xai_explainer = GradCAMExplainer(self.model, config)
        
        self.disease_names = config.get("multi_task", {}).get("pathology_classes", [
            "clubbing", "cyanosis", "melanoma", "onychogryphosis", "onychomycosis", "psoriasis", "healthy"
        ])
        
        self.disease_info_db = {
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
            },
            "healthy": {
                "description": "Normal, clear nail matrix with healthy pinkish bed and intact lunula.",
                "prevention": "Maintain daily hand hygiene and regular moisturizing.",
                "treatment": "None required. Continue standard nail hygiene."
            }
        }

    def prepare_input_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Resizes, normalizes, and constructs RGB PyTorch input tensor."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0).to(self.device)

    def analyze_image_bgr(self, image_bgr: np.ndarray, finger_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes complete V2 Research Pipeline on an input BGR image array.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {"error": "Invalid image input"}
            
        # 1. Image Quality Assessment (IQA)
        iqa_results = self.iqa_engine.assess_quality(image_bgr)
        
        # 2. Automated Nail Segmentation
        seg_results = self.segmenter.process(image_bgr)
        cropped_nail = seg_results["cropped_roi"]
        
        # 3. Illumination & Color Normalization
        norm_nail = self.normalizer.normalize(cropped_nail)
        
        # 4. Input Tensor Construction
        input_tensor = self.prepare_input_tensor(norm_nail)
        
        # 5. Multi-Task Model Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        pathology_logits = output["pathology_logits"]
        polish_logits = output["polish_logits"]
        
        pathology_probs = F.softmax(pathology_logits, dim=1).cpu().numpy()[0]
        polish_prob = float(torch.sigmoid(polish_logits).item())
        
        predicted_idx = int(np.argmax(pathology_probs))
        predicted_disease = self.disease_names[predicted_idx] if predicted_idx < len(self.disease_names) else "healthy"
        
        # 6. Monte Carlo Dropout Confidence & Epistemic Uncertainty
        mc_probs, variance, confidence_pct = self.confidence_estimator.evaluate_mc_dropout(self.model, input_tensor)
        
        # 7. Explainable AI (GradCAM++)
        heatmap_dir = self.config.get("paths", {}).get("heatmaps_dir", "results/heatmaps")
        heatmap_save_path = os.path.join(heatmap_dir, f"xai_{finger_name or 'nail'}.jpg")
        xai_out = self.xai_explainer.explain(input_tensor, norm_nail, target_class_idx=predicted_idx, save_path=heatmap_save_path)
        
        is_healthy = (predicted_disease.lower() == "healthy")
        disease_prob_pct = float(round((1.0 - pathology_probs[-1]) * 100.0 if "healthy" in self.disease_names else pathology_probs[predicted_idx] * 100.0, 2))
        
        # Format confidences dictionary
        all_confidences = {
            name.capitalize(): float(round(pathology_probs[i] * 100.0, 2))
            for i, name in enumerate(self.disease_names) if name.lower() != "healthy"
        }
        
        info = self.disease_info_db.get(predicted_disease.lower(), self.disease_info_db["healthy"])
        
        result = {
            "healthy": is_healthy,
            "disease": predicted_disease,
            "disease_probability": disease_prob_pct,
            "disease_confidence": float(round(pathology_probs[predicted_idx] * 100.0, 2)),
            "all_confidences": all_confidences,
            "polish_detected": (polish_prob >= 0.70) if is_healthy else (polish_prob >= 0.85),
            "polish_confidence": float(round(polish_prob * 100.0, 2)),
            "info": info,
            "iqa": iqa_results,
            "segmentation_bbox": seg_results["bbox"],
            "uncertainty_variance": float(round(variance, 6)),
            "confidence_score": confidence_pct,
            "xai_heatmap_path": heatmap_save_path if os.path.exists(heatmap_save_path) else None
        }
        
        if finger_name:
            result["finger"] = finger_name
            
        return result
