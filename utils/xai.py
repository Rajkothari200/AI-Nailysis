"""
AI Nailysis V2 - Module 8: Explainable AI (XAI) Engine
======================================================
Implements GradCAM, GradCAM++, and EigenCAM to compute pixel-level gradient activations,
visualize model focus areas, and export high-contrast JET colormap visual overlays.
"""

from typing import Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger("XAIEngine")


class GradCAMExplainer:
    """
    Computes GradCAM, GradCAM++, and EigenCAM activation heatmaps.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch Multi-Task model instance.
            config: System configuration dictionary.
            target_layer: Specific layer to hook. If None, automatically selects last conv layer.
        """
        self.model = model
        self.config = config.get("xai", {})
        self.enabled = self.config.get("enabled", True)
        self.default_method = self.config.get("default_method", "gradcam_pp").lower()
        self.save_heatmaps = self.config.get("save_heatmaps", True)
        self.alpha_overlay = float(self.config.get("alpha_overlay", 0.5))
        
        self.gradients = None
        self.activations = None
        
        # Locate target convolutional layer
        self.target_layer = target_layer if target_layer is not None else self._find_target_layer()
        if self.target_layer:
            self._register_hooks()

    def _find_target_layer(self) -> Optional[nn.Module]:
        """Automatically locates the final 2D Convolutional layer in the model."""
        target = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                target = module
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, input_tensor: torch.Tensor, target_class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates standard GradCAM activation heatmap.
        """
        self.model.zero_grad()
        output_dict = self.model(input_tensor)
        logits = output_dict["pathology_logits"]
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())
            
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            # Fallback uniform activation map
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
            
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

    def generate_gradcam_pp(self, input_tensor: torch.Tensor, target_class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates GradCAM++ activation heatmap for refined spatial detail.
        """
        self.model.zero_grad()
        output_dict = self.model(input_tensor)
        logits = output_dict["pathology_logits"]
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())
            
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0] # [C, H, W]
        
        grad_2 = grads ** 2
        grad_3 = grads ** 3
        
        sum_acts = torch.sum(acts, dim=[1, 2], keepdim=True)
        alpha_denom = 2 * grad_2 + sum_acts * grad_3 + 1e-7
        alpha = grad_2 / alpha_denom
        
        weights = torch.sum(alpha * F.relu(grads), dim=[1, 2], keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        
        cam = F.relu(cam).cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

    def generate_eigencam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates EigenCAM heatmap using principal components of feature activations.
        """
        with torch.no_grad():
            self.model(input_tensor)
            
        if self.activations is None:
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
            
        acts = self.activations[0].cpu().numpy()  # [C, H, W]
        c, h, w = acts.shape
        reshaped = acts.reshape(c, h * w).T  # [H*W, C]
        
        # PCA computation via SVD
        reshaped = reshaped - np.mean(reshaped, axis=0)
        _, _, vh = np.linalg.svd(reshaped, full_matrices=False)
        first_comp = vh[0, :]
        
        cam = np.dot(reshaped, first_comp).reshape(h, w)
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

    def create_heatmap_overlay(self, original_bgr: np.ndarray, cam_mask: np.ndarray) -> np.ndarray:
        """
        Resizes activation mask and blends JET colormap over the original BGR image.
        """
        h, w = original_bgr.shape[:2]
        resized_cam = cv2.resize(cam_mask, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(original_bgr, 1.0 - self.alpha_overlay, heatmap, self.alpha_overlay, 0)
        return overlay

    def explain(self, input_tensor: torch.Tensor, original_bgr: np.ndarray, method: Optional[str] = None, target_class_idx: Optional[int] = None, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs XAI generator and returns heatmaps and overlaid images.
        """
        if not self.enabled:
            return {"heatmap_overlay": original_bgr, "cam_mask": None}

        selected_method = (method or self.default_method).lower()
        
        if selected_method == "gradcam":
            cam = self.generate_gradcam(input_tensor, target_class_idx)
        elif selected_method == "gradcam_pp":
            cam = self.generate_gradcam_pp(input_tensor, target_class_idx)
        elif selected_method == "eigencam":
            cam = self.generate_eigencam(input_tensor)
        else:
            cam = self.generate_gradcam_pp(input_tensor, target_class_idx)
            
        overlay = self.create_heatmap_overlay(original_bgr, cam)
        
        if save_path and self.save_heatmaps:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, overlay)
            logger.info(f"Saved GradCAM heatmap to: {save_path}")
            
        return {
            "heatmap_overlay": overlay,
            "cam_mask": cam,
            "method": selected_method
        }
