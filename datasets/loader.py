"""
AI Nailysis V2 - PyTorch Custom Dataset & DataLoader Engine
============================================================
Defines PyTorch Dataset wrappers with Albumentations integration for clinical nail image batches.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.augmentations import NailAugmentor
from utils.logger import get_logger

logger = get_logger("DatasetLoader")


class AINailysisDataset(Dataset):
    """
    PyTorch Dataset for multi-task clinical nail learning.
    """
    def __init__(self, image_paths: List[str], labels: List[Dict[str, Any]], config: Dict[str, Any], is_training: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.is_training = is_training
        
        self.augmentor = NailAugmentor(config)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        image_bgr = cv2.imread(img_path)
        
        if image_bgr is None:
            # Fallback zero image
            img_size = tuple(self.config.get("augmentation", {}).get("image_size", [224, 224]))
            image_bgr = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = self.augmentor.transform_image(img_rgb, is_training=self.is_training)
        
        item_labels = self.labels[idx]
        target_dict = {
            "pathology": torch.tensor(item_labels.get("pathology", 6), dtype=torch.long),  # Default Healthy
            "color": torch.tensor(item_labels.get("color", 0), dtype=torch.long),
            "surface": torch.tensor(item_labels.get("surface", 0), dtype=torch.long),
            "polish": torch.tensor(item_labels.get("polish", 0.0), dtype=torch.float),
            "quality": torch.tensor(item_labels.get("quality", 1.0), dtype=torch.float)
        }
        
        return tensor_img, target_dict


def create_dataloaders(train_paths: List[str], train_labels: List[Dict[str, Any]], val_paths: List[str], val_labels: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoader instances for training and validation sets.
    """
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 2))
    
    train_dataset = AINailysisDataset(train_paths, train_labels, config, is_training=True)
    val_dataset = AINailysisDataset(val_paths, val_labels, config, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
