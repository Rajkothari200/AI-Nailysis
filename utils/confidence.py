"""
AI Nailysis V2 - Module 9: Confidence Estimation & Uncertainty Quantification
================================================================================
Implements Monte Carlo (MC) Dropout sampling for epistemic variance estimation,
Temperature Scaling calibration, and normalized Shannon Entropy confidence scoring.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger("ConfidenceEstimator")


def enable_dropout_at_test_time(model: nn.Module):
    """Enforces dropout layers to remain active during test-time evaluation."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


class ConfidenceEstimator:
    """
    Evaluates prediction confidence, epistemic variance (via MC Dropout), and calibrated probabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("confidence", {})
        self.num_samples = int(self.config.get("mc_dropout_samples", 20))
        self.temperature = float(self.config.get("temperature_scaling", 1.2))
        self.entropy_threshold = float(self.config.get("entropy_threshold", 0.70))

    def apply_temperature_scaling(self, logits: torch.Tensor, temp: Optional[float] = None) -> torch.Tensor:
        """
        Calibrates unnormalized logit outputs using Temperature parameter T.
        """
        t = temp if temp is not None else self.temperature
        return logits / float(t)

    def calculate_shannon_entropy(self, probs: np.ndarray) -> float:
        """
        Calculates normalized Shannon Entropy H(p) in range [0, 1].
        """
        num_classes = len(probs)
        if num_classes <= 1:
            return 0.0
            
        # Add epsilon for numerical stability
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(num_classes)
        return float(entropy / max_entropy)

    def evaluate_mc_dropout(self, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float, float]:
        """
        Executes N Monte Carlo Dropout stochastic forward passes.
        
        Returns:
            Tuple of (mean_probabilities, epistemic_variance_score, confidence_percentage)
        """
        model.eval()
        enable_dropout_at_test_time(model)
        
        sample_probs = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = model(input_tensor)
                logits = out["pathology_logits"]
                scaled_logits = self.apply_temperature_scaling(logits)
                probs = F.softmax(scaled_logits, dim=1).cpu().numpy()[0]
                sample_probs.append(probs)
                
        sample_probs = np.array(sample_probs)  # [N, num_classes]
        
        # Predictive Mean
        mean_probs = np.mean(sample_probs, axis=0)
        
        # Epistemic Uncertainty Variance across samples
        variance = float(np.mean(np.var(sample_probs, axis=0)))
        
        # Entropy
        norm_entropy = self.calculate_shannon_entropy(mean_probs)
        
        # Composite Confidence Percentage
        confidence_pct = max(0.0, min(100.0, (1.0 - norm_entropy) * (1.0 - variance * 2.0) * 100.0))
        
        return mean_probs, variance, round(confidence_pct, 2)
