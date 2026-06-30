"""
Model Export and Serialization
==============================
Exports trained models to production-ready formats:
1. TorchScript (JIT trace and script)
2. ONNX (Open Neural Network Exchange)

Supports both the Attention U-Net (Segmentation) and BUS-XAINet (Classification).
"""

import os
import sys
import torch
from typing import Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import AttentionUNet
from models.bus_xainet import BUSXAINet


def export_unet(checkpoint_path: Optional[str] = None, output_dir: str = "outputs/models") -> Tuple[str, str]:
    """Exports Attention U-Net model to TorchScript and ONNX."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Instantiate model
    model = AttentionUNet(in_channels=1, out_channels=1, pretrained=False)
    
    # Load weights if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading U-Net weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
    model.eval()
    
    # Dummy input (1, 1, 256, 256)
    dummy_input = torch.randn(1, 1, 256, 256)
    
    # 1. TorchScript Trace
    ts_path = os.path.join(output_dir, "unet_production.pt")
    traced_model = torch.jit.trace(model, dummy_input, check_trace=False)
    traced_model.save(ts_path)
    print(f"[Exported] U-Net TorchScript -> {ts_path}")
    
    # 2. ONNX Export
    onnx_path = os.path.join(output_dir, "unet_production.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["segmentation_mask"],
            dynamic_axes={"input_image": {0: "batch_size"}, "segmentation_mask": {0: "batch_size"}}
        )
        print(f"[Exported] U-Net ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[WARNING] U-Net ONNX export failed: {e}. TorchScript was exported successfully.")
        onnx_path = ""
        
    return ts_path, onnx_path


def export_bus_xainet(checkpoint_path: Optional[str] = None, output_dir: str = "outputs/models") -> Tuple[str, str]:
    """Exports BUS-XAINet model to TorchScript and ONNX."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Instantiate model
    model = BUSXAINet(num_classes=3, pretrained=False)
    
    # Load weights if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading BUS-XAINet weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
    model.eval()
    
    # Dummy input (1, 2, 224, 224) for 2-channel classifier
    dummy_input = torch.randn(1, 2, 224, 224)
    
    # 1. TorchScript Trace
    ts_path = os.path.join(output_dir, "bus_xainet_production.pt")
    # Tracing is preferred for hybrid architectures to freeze control flow
    traced_model = torch.jit.trace(model, dummy_input, check_trace=False)
    traced_model.save(ts_path)
    print(f"[Exported] BUS-XAINet TorchScript -> {ts_path}")
    
    # 2. ONNX Export
    onnx_path = os.path.join(output_dir, "bus_xainet_production.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["logits"],
            dynamic_axes={"input_image": {0: "batch_size"}, "logits": {0: "batch_size"}}
        )
        print(f"[Exported] BUS-XAINet ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[WARNING] BUS-XAINet ONNX export failed: {e}. TorchScript was exported successfully.")
        onnx_path = ""
        
    return ts_path, onnx_path


if __name__ == "__main__":
    print("=== Starting Model Export Pipeline ===")
    
    # Export with dummy weights if no checkpoints are found
    unet_ts, unet_onnx = export_unet()
    xainet_ts, xainet_onnx = export_bus_xainet()
    
    print("\n=== Model Export Completed successfully! ===")
