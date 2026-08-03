"""
AI Nailysis V2 - Module 3: Illumination & Color Normalization Engine
=======================================================================
Provides color and illuminant normalization algorithms (CLAHE, Gray World,
White Balance, Gamma Correction, Histogram Equalization) to standardize skin and
nail bed appearances across varying camera sensors and ambient light conditions.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("ColorNormEngine")


class ColorNormalizer:
    """
    Standardizes color, contrast, and illumination across input nail images.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing color_normalization section of config.yaml.
        """
        self.config = config.get("color_normalization", {})
        self.enabled = self.config.get("enabled", True)
        self.default_method = self.config.get("default_method", "clahe").lower()
        
        # Method parameters
        clahe_cfg = self.config.get("clahe", {})
        self.clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
        self.tile_grid_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        
        gamma_cfg = self.config.get("gamma", {})
        self.gamma_value = float(gamma_cfg.get("gamma_value", 1.2))
        
        wb_cfg = self.config.get("white_balance", {})
        self.minkowski_p = int(wb_cfg.get("minkowski_p", 6))

    def apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization to the L-channel of LAB space.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_gray_world(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Gray World Algorithm for illuminant correction.
        Assumes average color of the scene is neutral gray.
        """
        b, g, r = cv2.split(image_bgr.astype(np.float32))
        mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
        
        gray_mean = (mean_b + mean_g + mean_r) / 3.0
        
        kb = gray_mean / (mean_b + 1e-6)
        kg = gray_mean / (mean_g + 1e-6)
        kr = gray_mean / (mean_r + 1e-6)
        
        b = np.clip(b * kb, 0, 255)
        g = np.clip(g * kg, 0, 255)
        r = np.clip(r * kr, 0, 255)
        
        return cv2.merge([b, g, r]).astype(np.uint8)

    def apply_white_balance(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Shades-of-Gray White Balance using Minkowski p-norm.
        """
        img_float = image_bgr.astype(np.float32)
        p = self.minkowski_p
        
        # Calculate p-norm per channel
        norm_b = np.power(np.mean(np.power(img_float[:, :, 0], p)), 1.0 / p)
        norm_g = np.power(np.mean(np.power(img_float[:, :, 1], p)), 1.0 / p)
        norm_r = np.power(np.mean(np.power(img_float[:, :, 2], p)), 1.0 / p)
        
        avg_norm = (norm_b + norm_g + norm_r) / 3.0
        
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (avg_norm / (norm_b + 1e-6)), 0, 255)
        img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (avg_norm / (norm_g + 1e-6)), 0, 255)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (avg_norm / (norm_r + 1e-6)), 0, 255)
        
        return img_float.astype(np.uint8)

    def apply_gamma_correction(self, image_bgr: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Applies Non-linear Gamma Correction.
        """
        g = gamma if gamma is not None else self.gamma_value
        inv_gamma = 1.0 / g
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image_bgr, table)

    def apply_histogram_equalization(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Histogram Equalization on the Y-channel in YUV color space.
        """
        yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def normalize(self, image_bgr: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Applies selected illumination/color normalization technique.
        
        Args:
            image_bgr: OpenCV image array in BGR format.
            method: Overrides default configured method ("clahe", "gray_world", "white_balance", "gamma", "equalize", "none").
            
        Returns:
            Normalized OpenCV image array in BGR format.
        """
        if not self.enabled or image_bgr is None or image_bgr.size == 0:
            return image_bgr

        selected = (method or self.default_method).lower()
        
        if selected == "clahe":
            return self.apply_clahe(image_bgr)
        elif selected == "gray_world":
            return self.apply_gray_world(image_bgr)
        elif selected == "white_balance":
            return self.apply_white_balance(image_bgr)
        elif selected == "gamma":
            return self.apply_gamma_correction(image_bgr)
        elif selected == "equalize":
            return self.apply_histogram_equalization(image_bgr)
        elif selected == "none":
            return image_bgr
        else:
            logger.warning(f"Unknown color normalization method '{selected}'. Falling back to CLAHE.")
            return self.apply_clahe(image_bgr)
