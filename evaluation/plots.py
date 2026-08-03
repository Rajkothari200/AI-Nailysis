"""
AI Nailysis V2 - Module 10: Research Visualization & Plotting Engine
======================================================================
Generates publication-ready figures for Confusion Matrices, ROC Curves,
Precision-Recall Curves, and Expected Calibration Error (ECE) Diagrams.
"""

from typing import Dict, Any, List, Optional
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.logger import get_logger

logger = get_logger("PlotsEngine")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, title: str = "Confusion Matrix"):
    """Generates and saves a high-contrast normalized confusion matrix heatmap."""
    plt.figure(figsize=(8, 6), dpi=300)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def plot_roc_curves(fpr_dict: Dict[str, np.ndarray], tpr_dict: Dict[str, np.ndarray], auc_dict: Dict[str, float], save_path: str):
    """Generates and saves multi-class One-vs-Rest Receiver Operating Characteristic (ROC) curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    for cls_name in fpr_dict:
        plt.plot(fpr_dict[cls_name], tpr_dict[cls_name], label=f"{cls_name.capitalize()} (AUC = {auc_dict[cls_name]:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Multi-Class One-vs-Rest ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curves plot to {save_path}")


def plot_calibration_curve(prob_true: np.ndarray, prob_pred: np.ndarray, ece: float, save_path: str):
    """Generates and saves Reliability Calibration Diagram."""
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(prob_pred, prob_true, "s-", label=f"Model Calibration (ECE = {ece*100:.2f}%)")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    
    plt.xlabel("Mean Predicted Confidence", fontsize=12)
    plt.ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    plt.title("Reliability Calibration Diagram", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved calibration curve plot to {save_path}")
