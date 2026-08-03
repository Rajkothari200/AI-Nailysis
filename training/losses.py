"""
AI Nailysis V2 - Combined Multi-Task Loss Functions
====================================================
Computes weighted composite loss across pathology, color, surface, polish, and image quality heads.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of classification and regression loss heads.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        weights_cfg = config.get("multi_task", {}).get("loss_weights", {})
        self.w_pathology = float(weights_cfg.get("pathology", 1.0))
        self.w_color = float(weights_cfg.get("color", 0.5))
        self.w_surface = float(weights_cfg.get("surface", 0.5))
        self.w_polish = float(weights_cfg.get("polish", 0.8))
        self.w_quality = float(weights_cfg.get("quality", 0.3))
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.SmoothL1Loss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_pathology = self.ce_loss(predictions["pathology_logits"], targets["pathology"])
        loss_color = self.ce_loss(predictions["color_logits"], targets["color"])
        loss_surface = self.ce_loss(predictions["surface_logits"], targets["surface"])
        loss_polish = self.bce_loss(predictions["polish_logits"], targets["polish"])
        loss_quality = self.mse_loss(predictions["quality_score"], targets["quality"])
        
        total_loss = (
            self.w_pathology * loss_pathology +
            self.w_color * loss_color +
            self.w_surface * loss_surface +
            self.w_polish * loss_polish +
            self.w_quality * loss_quality
        )
        
        return {
            "total_loss": total_loss,
            "loss_pathology": loss_pathology,
            "loss_color": loss_color,
            "loss_surface": loss_surface,
            "loss_polish": loss_polish,
            "loss_quality": loss_quality
        }
