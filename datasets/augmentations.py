"""
AI Nailysis V2 - Module 4: Advanced Research Data Augmentation Engine
========================================================================
Implements Albumentations pipelines with affine transforms, color jittering,
elastic deformations, coarse dropout, and batch-level CutMix & MixUp strategies.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.logger import get_logger

logger = get_logger("AugmentationEngine")


class NailAugmentor:
    """
    Albumentations pipeline and batch-level convex combination augmentations (MixUp/CutMix).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("augmentation", {})
        self.img_size = tuple(self.config.get("image_size", [224, 224]))
        
        self.train_transform = self._build_train_pipeline()
        self.eval_transform = self._build_eval_pipeline()

    def _build_train_pipeline() -> A.Compose:
        """Constructs stochastic research training augmentation pipeline."""
        deg = self.config.get("random_rotate_deg", 30)
        bright = self.config.get("brightness_limit", 0.2)
        contrast = self.config.get("contrast_limit", 0.2)
        
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=deg, p=0.7),
            A.ColorJitter(brightness=bright, contrast=contrast, saturation=0.2, hue=0.1, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=self.config.get("coarse_dropout_max_holes", 8),
                max_height=self.config.get("coarse_dropout_max_height", 16),
                max_width=self.config.get("coarse_dropout_max_width", 16),
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _build_eval_pipeline() -> A.Compose:
        """Constructs evaluation/inference normalization pipeline."""
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def transform_image(self, image_rgb: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """
        Transforms an RGB image array into a PyTorch tensor.
        """
        pipeline = self.train_transform if is_training else self.eval_transform
        augmented = pipeline(image=image_rgb)
        return augmented["image"]


def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp regularization across a mini-batch.
    x' = lambda * x_i + (1 - lambda) * x_j
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def apply_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix regularization across a mini-batch.
    Replaces a random bounding box in x_i with a patch from x_j.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
