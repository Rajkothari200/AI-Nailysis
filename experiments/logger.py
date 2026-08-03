"""
AI Nailysis V2 - Module 11: Experiment Tracking & Logging Engine
===================================================================
Automatically manages timestamped experiment run directories, writes hyperparameter snapshots,
tracks epoch metrics in CSV format, exports metric summaries in JSON, and archives model checkpoints.
"""

from typing import Dict, Any, Optional
import os
import json
import csv
import time
import yaml
from utils.logger import get_logger

logger = get_logger("ExperimentTracker")


class ExperimentTracker:
    """
    Automated experiment logger managing directory structures, CSV logs, and JSON artifacts.
    """
    def __init__(self, config: Dict[str, Any], experiment_name: str = "wacv_experiment"):
        self.config = config
        self.base_dir = config.get("paths", {}).get("experiments_dir", "experiments")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"{experiment_name}_{timestamp}")
        self.weights_dir = os.path.join(self.run_dir, "weights")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Save snapshot of config.yaml
        self._save_config_snapshot()
        
        # Initialize history CSV file
        self.csv_path = os.path.join(self.run_dir, "history.csv")
        self.csv_initialized = False

    def _save_config_snapshot(self):
        """Writes exact replica of system config.yaml into run directory."""
        snapshot_path = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved experiment configuration snapshot to {snapshot_path}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Appends epoch metrics row to CSV file.
        """
        row_dict = {"epoch": epoch, **metrics}
        fieldnames = list(row_dict.keys())
        
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def save_metrics_summary(self, summary: Dict[str, Any]):
        """
        Saves final metrics summary JSON file.
        """
        summary_path = os.path.join(self.run_dir, "final_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved final experiment summary to {summary_path}")

    def get_best_weight_path(self, filename: str = "best_model.pth") -> str:
        """Returns target filepath for saving best model checkpoint."""
        return os.path.join(self.weights_dir, filename)
