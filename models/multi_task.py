"""
AI Nailysis V2 - Module 7: Multi-Task Neural Network Architecture
===================================================================
Defines a multi-task learning model combining a shared backbone, attention module,
and specialized prediction heads for pathologies, color abnormalities, surface dystrophies,
nail polish detection, image quality regression, and confidence estimation.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import HybridBackbone
from models.attention import FeatureAttentionFactory
from utils.logger import get_logger

logger = get_logger("MultiTaskModel")


class MultiTaskAINailysisModel(nn.Module):
    """
    Research-grade Multi-Task Network for Clinical Nail Assessment.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Shared Backbone
        self.backbone = HybridBackbone(config)
        feature_dim = self.backbone.feature_dim
        
        # Attention Layer
        self.attention = FeatureAttentionFactory(feature_dim, config)
        
        # Multi-task heads configuration
        mt_cfg = config.get("multi_task", {})
        num_pathologies = int(mt_cfg.get("num_pathology_classes", 7))
        num_color_classes = int(mt_cfg.get("color_abnormalities", 3))
        num_surface_classes = int(mt_cfg.get("surface_dystrophies", 3))
        drop_rate = float(config.get("backbone", {}).get("drop_rate", 0.3))
        
        # Head 1: Pathology Classification Head (Clubbing, Cyanosis, Melanoma, Onychogryphosis, Onychomycosis, Psoriasis, Healthy)
        self.pathology_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_pathologies)
        )
        
        # Head 2: Color Abnormality Head (Normal, Discolored, Melanonychia)
        self.color_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_color_classes)
        )
        
        # Head 3: Surface Dystrophy Head (Smooth, Pitted/Ridged, Thickened)
        self.surface_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_surface_classes)
        )
        
        # Head 4: Polish / Art Detection Head (Binary)
        self.polish_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Head 5: Image Quality Regression Head
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 6: Confidence / Variance Head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding multi-task predictions.
        
        Args:
            x: Input image tensor batch [B, C, H, W]
            
        Returns:
            Dictionary of tensor predictions for each head.
        """
        # Shared Backbone Feature Extraction
        features = self.backbone(x)
        
        # Attention Recalibration
        features = self.attention(features)
        
        # Prediction Heads
        pathology_logits = self.pathology_head(features)
        color_logits = self.color_head(features)
        surface_logits = self.surface_head(features)
        polish_logits = self.polish_head(features).squeeze(-1)
        quality_score = self.quality_head(features).squeeze(-1)
        confidence_score = self.confidence_head(features).squeeze(-1)
        
        return {
            "pathology_logits": pathology_logits,
            "color_logits": color_logits,
            "surface_logits": surface_logits,
            "polish_logits": polish_logits,
            "quality_score": quality_score,
            "confidence_score": confidence_score,
            "features": features
        }
