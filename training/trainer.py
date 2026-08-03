"""
AI Nailysis V2 - PyTorch Multi-Task Trainer
============================================
Handles model optimization, mixed-precision training (torch.cuda.amp),
validation loops, and automated checkpoint archiving.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.losses import MultiTaskLoss
from experiments.logger import ExperimentTracker
from utils.logger import get_logger

logger = get_logger("Trainer")


class MultiTaskTrainer:
    """
    Research trainer supporting PyTorch AMP, Cosine Annealing, and Experiment Tracking.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], tracker: ExperimentTracker):
        self.model = model
        self.config = config
        self.tracker = tracker
        
        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 50))
        self.lr = float(train_cfg.get("lr", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.005))
        self.amp_enabled = bool(config.get("system", {}).get("mixed_precision", True))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("system", {}).get("device") == "cuda" else "cpu")
        self.model.to(self.device)
        
        self.criterion = MultiTaskLoss(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled and self.device.type == "cuda")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                preds = self.model(images)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict["total_loss"]
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(dataloader))
        return {"train_loss": avg_loss}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    preds = self.model(images)
                    loss_dict = self.criterion(preds, targets)
                    loss = loss_dict["total_loss"]
                    
                total_loss += loss.item()
                
        avg_loss = total_loss / max(1, len(dataloader))
        return {"val_loss": avg_loss}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Executes complete multi-epoch training loop.
        """
        best_val_loss = float("inf")
        logger.info(f"Starting training run for {self.epochs} epochs on device '{self.device}'...")
        
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            
            combined = {**train_metrics, **val_metrics, "lr": self.optimizer.param_groups[0]["lr"]}
            self.tracker.log_epoch(epoch, combined)
            
            logger.info(f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_path = self.tracker.get_best_weight_path("best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best validation model saved to: {save_path}")
