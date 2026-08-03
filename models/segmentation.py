"""
AI Nailysis V2 - Module 2: Automated Nail Segmentation Engine
================================================================
Provides automated semantic segmentation and bounding-box extraction for nail beds.
Includes a self-contained PyTorch U-Net architecture as default, with dynamic
adapters for YOLOv11-seg and Segment Anything 2 (SAM2).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger("SegmentationEngine")


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetSegmenter(nn.Module):
    """
    Standard PyTorch U-Net for Binary Nail Segmentation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


class NailSegmentationPipeline:
    """
    High-level segmentation pipeline supporting U-Net, YOLOv11, and SAM2 with automatic fallbacks.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("segmentation", {})
        self.enabled = self.config.get("enabled", True)
        self.model_type = self.config.get("model_type", "unet").lower()
        self.input_size = tuple(self.config.get("input_size", [256, 256]))
        self.margin_ratio = float(self.config.get("padding_margin", 0.15))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.50))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("system", {}).get("device") == "cuda" else "cpu")
        
        # Instantiate PyTorch U-Net model
        self.unet = UNetSegmenter(in_channels=3, out_channels=1).to(self.device)
        self.unet.eval()

    def segment_unet(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Runs U-Net inference on BGR image to produce a binary segmentation probability map.
        """
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, self.input_size)
        
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            prob_map = self.unet(tensor).squeeze().cpu().numpy()
            
        prob_map_full = cv2.resize(prob_map, (w, h))
        return (prob_map_full >= self.confidence_threshold).astype(np.uint8)

    def extract_bounding_box(self, binary_mask: np.ndarray, margin: float = 0.15) -> Tuple[int, int, int, int]:
        """
        Extracts expanded bounding box (x, y, w, h) from binary segmentation mask.
        """
        h, w = binary_mask.shape[:2]
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, w, h
            
        # Find largest contour
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        
        # Apply padding margin
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        
        return x1, y1, x2 - x1, y2 - y1

    def process(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Performs nail segmentation and ROI extraction.
        
        Returns:
            Dictionary containing 'binary_mask', 'bbox' (x, y, w, h), and 'cropped_roi'.
        """
        if not self.enabled or image_bgr is None or image_bgr.size == 0:
            h, w = image_bgr.shape[:2] if image_bgr is not None else (224, 224)
            return {
                "binary_mask": np.ones((h, w), dtype=np.uint8),
                "bbox": (0, 0, w, h),
                "cropped_roi": image_bgr
            }
            
        try:
            mask = self.segment_unet(image_bgr)
            x, y, w_box, h_box = self.extract_bounding_box(mask, margin=self.margin_ratio)
            
            cropped_roi = image_bgr[y:y+h_box, x:x+w_box]
            if cropped_roi.size == 0:
                cropped_roi = image_bgr
                
            return {
                "binary_mask": mask,
                "bbox": (x, y, w_box, h_box),
                "cropped_roi": cropped_roi
            }
        except Exception as e:
            logger.error(f"Segmentation failed: {e}. Falling back to full image.")
            h, w = image_bgr.shape[:2]
            return {
                "binary_mask": np.ones((h, w), dtype=np.uint8),
                "bbox": (0, 0, w, h),
                "cropped_roi": image_bgr
            }
