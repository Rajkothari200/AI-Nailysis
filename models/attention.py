"""
AI Nailysis V2 - Module 6: Advanced Attention Modules
======================================================
Provides modular PyTorch attention layers: CBAM (Channel + Spatial),
SE Block (Squeeze-and-Excitation), Coordinate Attention, and Multi-Head Self-Attention.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger("AttentionModule")


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    Applies channel-wise recalibration.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(8, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(8, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CoordinateAttention(nn.Module):
    """Coordinate Attention for Efficient Feature Localization"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(8, channels // reduction_ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, reduced, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(reduced, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduced, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return identity * a_h * a_w


class FeatureAttentionFactory(nn.Module):
    """
    Attention module wrapper allowing configuration-driven switching.
    """
    def __init__(self, channels: int, config: Dict[str, Any]):
        super().__init__()
        attn_cfg = config.get("attention", {})
        self.enabled = attn_cfg.get("enabled", True)
        self.attn_type = attn_cfg.get("type", "cbam").lower()
        reduction = int(attn_cfg.get("reduction_ratio", 16))
        
        if not self.enabled or self.attn_type == "none":
            self.module = nn.Identity()
        elif self.attn_type == "cbam":
            self.module = CBAMBlock(channels, reduction_ratio=reduction)
        elif self.attn_type == "se":
            self.module = SEBlock(channels, reduction_ratio=reduction)
        elif self.attn_type == "coord":
            self.module = CoordinateAttention(channels, reduction_ratio=reduction)
        else:
            logger.warning(f"Unknown attention type '{self.attn_type}'. Defaulting to CBAM.")
            self.module = CBAMBlock(channels, reduction_ratio=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If feature is 1D embedding [B, C], convert to [B, C, 1, 1] for spatial attention
        if len(x.shape) == 2:
            x_4d = x.unsqueeze(-1).unsqueeze(-1)
            out_4d = self.module(x_4d)
            return out_4d.squeeze(-1).squeeze(-1)
        return self.module(x)
