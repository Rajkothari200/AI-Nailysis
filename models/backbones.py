"""
AI Nailysis V2 - Module 5: Hybrid Backbone Feature Extractor
==============================================================
Provides factory loading for modern vision backbones: EfficientNetV2,
ConvNeXt, Swin Transformer, MobileViT, and Vision Transformer (ViT).
Includes fail-safe fallback options for pure PyTorch and torchvision.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("BackboneFactory")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class PurePyTorchConvNet(nn.Module):
    """Deep Convolutional Feature Extractor built with standard PyTorch layers."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridBackbone(nn.Module):
    """
    Unified feature extractor backbone with configurable model selection.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.get("backbone", {})
        self.model_name = self.config.get("model_name", "convnext").lower()
        self.pretrained = self.config.get("pretrained", True)
        self.drop_rate = float(self.config.get("drop_rate", 0.3))
        
        timm_names = self.config.get("timm_names", {
            "convnext": "convnext_small",
            "efficientnetv2": "tf_efficientnetv2_s",
            "swin": "swin_tiny_patch4_window7_224",
            "mobilevit": "mobilevit_s",
            "vit": "vit_base_patch16_224"
        })
        
        target_timm_name = timm_names.get(self.model_name, "convnext_small")
        
        if HAS_TIMM:
            logger.info(f"Building hybrid backbone via timm: {self.model_name} ('{target_timm_name}')")
            try:
                self.backbone = timm.create_model(
                    target_timm_name,
                    pretrained=self.pretrained,
                    num_classes=0,
                    drop_rate=self.drop_rate
                )
                self.feature_dim = self.backbone.num_features
                return
            except Exception as e:
                logger.warning(f"Failed to create timm model '{target_timm_name}': {e}.")

        if HAS_TORCHVISION:
            logger.info("Using torchvision ResNet50 backbone.")
            try:
                resnet = tv_models.resnet50(pretrained=self.pretrained)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                self.feature_dim = 2048
                return
            except Exception as e:
                logger.warning(f"Failed to create torchvision model: {e}.")

        logger.info("Building PyTorch ConvNet backbone.")
        self.backbone = PurePyTorchConvNet(in_channels=3, feature_dim=512)
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts global feature embeddings from input image tensor.
        """
        feats = self.backbone(x)
        if len(feats.shape) > 2:
            feats = feats.mean(dim=[-2, -1])
        return feats
