"""
AI Nailysis V2 - Module 15: Model Export & Acceleration Engine
===============================================================
Exports PyTorch Multi-Task models to ONNX runtime format with dynamic batching,
and generates TensorRT build directives for high-performance edge deployment.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("Exporter")


class ModelExporter:
    """
    Handles ONNX export and TensorRT directive creation.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config.get("export", {})
        self.onnx_cfg = self.config.get("onnx", {})
        self.trt_cfg = self.config.get("tensorrt", {})
        
        self.opset_version = int(self.onnx_cfg.get("opset_version", 17))
        self.default_onnx_path = self.onnx_cfg.get("export_path", "weights/ai_nailysis_v2.onnx")

    def export_onnx(self, output_path: Optional[str] = None, dummy_input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> str:
        """
        Exports PyTorch model to ONNX format with dynamic batch and spatial dimensions.
        
        Args:
            output_path: Target path for .onnx file.
            dummy_input_shape: Dummy tensor dimensions [B, C, H, W]
            
        Returns:
            Absolute filepath to the saved ONNX file.
        """
        target_path = output_path or self.default_onnx_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        self.model.eval()
        dummy_input = torch.randn(*dummy_input_shape, device="cpu")
        
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "pathology_logits": {0: "batch_size"},
            "color_logits": {0: "batch_size"},
            "surface_logits": {0: "batch_size"},
            "polish_logits": {0: "batch_size"},
            "quality_score": {0: "batch_size"},
            "confidence_score": {0: "batch_size"}
        }
        
        input_names = ["input"]
        output_names = [
            "pathology_logits", "color_logits", "surface_logits",
            "polish_logits", "quality_score", "confidence_score"
        ]
        
        logger.info(f"Exporting PyTorch model to ONNX (opset {self.opset_version}): {target_path}")
        
        torch.onnx.export(
            self.model.cpu(),
            dummy_input,
            target_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"ONNX export successful: {target_path}")
        return target_path

    def generate_tensorrt_script(self, onnx_path: str, script_path: str = "weights/build_tensorrt.sh"):
        """
        Generates shell script directive for TensorRT trtexec compilation.
        """
        precision = self.trt_cfg.get("precision", "fp16")
        workspace = self.trt_cfg.get("workspace_size_mb", 2048)
        engine_path = onnx_path.replace(".onnx", ".engine")
        
        script_content = f"""#!/bin/bash
# TensorRT Compilation Script for AI Nailysis V2
trtexec --onnx={onnx_path} \\
        --saveEngine={engine_path} \\
        --{precision} \\
        --memPoolSize=workspace:{workspace}M \\
        --minShapes=input:1x3x224x224 \\
        --optShapes=input:4x3x224x224 \\
        --maxShapes=input:16x3x224x224
echo "[OK] TensorRT Engine compiled: {engine_path}"
"""
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        logger.info(f"Generated TensorRT build script at {script_path}")
