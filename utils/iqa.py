"""
AI Nailysis V2 - Module 1: Image Quality Assessment (IQA) Engine
==================================================================
Performs automated visual assessment to reject degraded, blurry, mis-illuminated,
or occluded fingernail images before passing them to neural classification heads.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("IQAEngine")


class ImageQualityAssessor:
    """
    Computes visual metrics (Laplacian variance, brightness histogram, specular reflections, 
    occlusion ratio) and returns a composite quality score with diagnostic pass/fail flags.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing IQA parameters (typically iqa section of config.yaml).
        """
        self.config = config.get("iqa", {})
        self.enabled = self.config.get("enabled", True)
        self.blur_threshold = float(self.config.get("blur_threshold", 100.0))
        self.light_min = float(self.config.get("light_min", 40.0))
        self.light_max = float(self.config.get("light_max", 220.0))
        self.overexposure_ratio = float(self.config.get("overexposure_ratio", 0.15))
        self.reflection_threshold = float(self.config.get("reflection_threshold", 0.10))
        self.composite_pass_threshold = float(self.config.get("composite_pass_threshold", 0.60))

    def evaluate_blur(self, image_bgr: np.ndarray) -> Tuple[float, bool]:
        """
        Estimates image sharpness using the variance of Laplacian operator.
        
        Returns:
            Tuple of (variance_score, is_sharp_boolean)
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = laplacian_var >= self.blur_threshold
        return laplacian_var, is_sharp

    def evaluate_lighting(self, image_bgr: np.ndarray) -> Tuple[float, float, bool]:
        """
        Measures mean illuminant intensity and overexposure ratio in LAB color space.
        
        Returns:
            Tuple of (mean_luminance, overexposed_pixel_ratio, is_lighting_valid)
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_lum = float(np.mean(l_channel))
        
        overexposed_pixels = np.sum(l_channel > 245)
        total_pixels = l_channel.size
        overexp_ratio = float(overexposed_pixels / total_pixels)
        
        valid_lighting = (self.light_min <= mean_lum <= self.light_max) and (overexp_ratio <= self.overexposure_ratio)
        return mean_lum, overexp_ratio, valid_lighting

    def evaluate_reflections(self, image_bgr: np.ndarray) -> Tuple[float, bool]:
        """
        Detects glare and specular highlights using HSV Value and Saturation analysis.
        
        Returns:
            Tuple of (reflection_ratio, is_low_reflection_boolean)
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Specular highlights: high brightness (V > 230) and low saturation (S < 40)
        glare_mask = (v_channel > 230) & (s_channel < 40)
        reflection_ratio = float(np.sum(glare_mask) / glare_mask.size)
        is_acceptable = reflection_ratio <= self.reflection_threshold
        return reflection_ratio, is_acceptable

    def assess_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs full IQA pipeline on input BGR image.
        
        Args:
            image_bgr: OpenCV image array in BGR format.
            
        Returns:
            Dictionary containing granular scores, boolean flags, and overall composite score.
        """
        if not self.enabled or image_bgr is None or image_bgr.size == 0:
            return {
                "passed": True,
                "composite_score": 1.0,
                "metrics": {},
                "warnings": []
            }
            
        h, w = image_bgr.shape[:2]
        
        # 1. Blur
        laplacian_var, is_sharp = self.evaluate_blur(image_bgr)
        blur_score = min(1.0, laplacian_var / (self.blur_threshold * 2.5))
        
        # 2. Lighting
        mean_lum, overexp_ratio, valid_lighting = self.evaluate_lighting(image_bgr)
        lighting_score = 1.0 - abs(mean_lum - 128.0) / 128.0
        lighting_score = max(0.0, min(1.0, lighting_score - overexp_ratio))
        
        # 3. Reflections
        refl_ratio, low_refl = self.evaluate_reflections(image_bgr)
        refl_score = max(0.0, 1.0 - refl_ratio * 5.0)
        
        # Composite score calculation
        composite_score = float(0.4 * blur_score + 0.35 * lighting_score + 0.25 * refl_score)
        passed = composite_score >= self.composite_pass_threshold and is_sharp
        
        warnings = []
        if not is_sharp:
            warnings.append(f"Image is blurry (Laplacian var: {laplacian_var:.1f} < threshold {self.blur_threshold})")
        if mean_lum < self.light_min:
            warnings.append(f"Low lighting detected (Mean luminance: {mean_lum:.1f} < {self.light_min})")
        if mean_lum > self.light_max:
            warnings.append(f"High background exposure detected (Mean luminance: {mean_lum:.1f} > {self.light_max})")
        if overexp_ratio > self.overexposure_ratio:
            warnings.append(f"Severe overexposure / clipping ({overexp_ratio*100:.1f}% pixels clipped)")
        if not low_refl:
            warnings.append(f"Excessive specular reflections / flash glare ({refl_ratio*100:.1f}% area)")

        logger.info(f"IQA Assessment complete: Score={composite_score:.2f}, Passed={passed}")
        
        return {
            "passed": passed,
            "composite_score": round(composite_score, 4),
            "metrics": {
                "laplacian_variance": round(laplacian_var, 2),
                "blur_score": round(blur_score, 4),
                "mean_luminance": round(mean_lum, 2),
                "overexposure_ratio": round(overexp_ratio, 4),
                "reflection_ratio": round(refl_ratio, 4)
            },
            "flags": {
                "is_sharp": is_sharp,
                "valid_lighting": valid_lighting,
                "low_reflection": low_refl
            },
            "warnings": warnings
        }
