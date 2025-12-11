"""Utility functions for device management and reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification. If "auto", automatically select best available device.
        
    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> dict[str, str]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information.
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = str(torch.cuda.device_count())
        info["current_cuda_device"] = str(torch.cuda.current_device())
    
    return info


def move_to_device(data: torch.Tensor | dict | list, device: torch.device) -> torch.Tensor | dict | list:
    """
    Move data to specified device.
    
    Args:
        data: Data to move to device.
        device: Target device.
        
    Returns:
        Data moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data
