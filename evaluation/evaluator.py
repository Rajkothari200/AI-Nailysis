"""
AI Nailysis V2 - Module 10: Research Evaluation Suite
======================================================
Computes comprehensive computer vision metrics: Confusion Matrix, ROC-AUC, PR Curves,
Accuracy, Precision, Recall, F1-Score, Specificity, Sensitivity, and Calibration Error.
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_calibration_curve
from utils.logger import get_logger

logger = get_logger("Evaluator")


class ModelEvaluator:
    """
    Computes research evaluation metrics for multi-task predictions.
    """
    def __init__(self, class_names: List[str], save_dir: str = "results/plots"):
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def calculate_specificity_sensitivity(self, cm: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes per-class Sensitivity (Recall) and Specificity (TN / (TN + FP)).
        """
        num_classes = len(self.class_names)
        sensitivity = {}
        specificity = {}
        
        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            
            sens = tp / (tp + fn + 1e-7)
            spec = tn / (tn + fp + 1e-7)
            
            cls_name = self.class_names[i]
            sensitivity[cls_name] = float(sens)
            specificity[cls_name] = float(spec)
            
        return sensitivity, specificity

    def calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculates Expected Calibration Error (ECE) and returns calibration bin curves.
        """
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        prob_true = []
        prob_pred = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
                
                prob_true.append(accuracy_in_bin)
                prob_pred.append(avg_confidence_in_bin)
                
        return float(ece), np.array(prob_true), np.array(prob_pred)

    def evaluate_predictions(self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "test") -> Dict[str, Any]:
        """
        Runs comprehensive metric evaluation on predicted probabilities.
        
        Args:
            y_true: True integer class labels [N]
            y_prob: Predicted probability array [N, num_classes]
            prefix: Prefix identifier for output plots
            
        Returns:
            Structured dictionary of all computed metrics.
        """
        y_pred = np.argmax(y_prob, axis=1)
        
        # 1. Basic metrics
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        
        # 2. Confusion Matrix & Specificity/Sensitivity
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        sensitivity_dict, specificity_dict = self.calculate_specificity_sensitivity(cm)
        
        # Plot CM
        cm_path = os.path.join(self.save_dir, f"{prefix}_confusion_matrix.png")
        plot_confusion_matrix(cm, self.class_names, cm_path, title=f"Confusion Matrix ({prefix.capitalize()})")
        
        # 3. One-vs-Rest ROC & AUC
        fpr_dict, tpr_dict, auc_dict = {}, {}, {}
        for i, cls_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            fpr_dict[cls_name] = fpr
            tpr_dict[cls_name] = tpr
            auc_dict[cls_name] = float(auc(fpr, tpr))
            
        mean_auc = float(np.mean(list(auc_dict.values())))
        
        # Plot ROC Curves
        roc_path = os.path.join(self.save_dir, f"{prefix}_roc_curves.png")
        plot_roc_curves(fpr_dict, tpr_dict, auc_dict, roc_path)
        
        # 4. Calibration (ECE)
        ece, p_true, p_pred = self.calculate_ece(y_true, y_prob)
        cal_path = os.path.join(self.save_dir, f"{prefix}_calibration_curve.png")
        plot_calibration_curve(p_true, p_pred, ece, cal_path)
        
        metrics_summary = {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(f1, 4),
            "mean_auc": round(mean_auc, 4),
            "expected_calibration_error": round(ece, 4),
            "auc_per_class": auc_dict,
            "sensitivity_per_class": sensitivity_dict,
            "specificity_per_class": specificity_dict
        }

        logger.info(f"Evaluation Complete [{prefix}]: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Mean-AUC={mean_auc:.4f}, ECE={ece:.4f}")
        return metrics_summary
