"""GPU utilities for device selection and management."""

import os
import platform
from typing import Literal, Optional

import torch

from crown_common.config import get_settings
from crown_common.logging import get_logger

logger = get_logger(__name__)

DeviceType = Literal["cuda", "mps", "cpu", "mlx"]


def has_mlx() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        return platform.system() == "Darwin" and platform.processor() == "arm"
    except ImportError:
        return False


def get_device(prefer_local: bool = True) -> torch.device:
    """
    Get the appropriate torch device based on availability and settings.
    
    Priority (when prefer_local=True, optimized for M4 Pro):
    1. Settings override (if not 'auto')
    2. MPS if available (Mac M-series)
    3. CUDA if available
    4. CPU fallback
    
    Priority (when prefer_local=False, cloud mode):
    1. Settings override (if not 'auto')
    2. CUDA if available
    3. MPS if available
    4. CPU fallback
    
    Args:
        prefer_local: Prioritize MPS over CUDA for local development
    
    Returns:
        torch.device: Selected device
    """
    settings = get_settings()
    
    if settings.device != "auto":
        device_type = settings.device
        logger.info("Using configured device", device=device_type)
    else:
        # Check if we're in local mode (default) or cloud mode
        local_mode = settings.compute_mode == "local" if hasattr(settings, "compute_mode") else prefer_local
        
        if local_mode:
            # Local mode: prioritize MPS for M-series Macs
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device_type = "mps"
                logger.info("MPS (Metal Performance Shaders) available - M4 Pro optimized")
                # Don't set as default - causes issues with DataLoader
                # Models will be explicitly moved to MPS
            elif torch.cuda.is_available():
                device_type = "cuda"
                logger.info(
                    "CUDA available",
                    device_count=torch.cuda.device_count(),
                    device_name=torch.cuda.get_device_name(0),
                )
            else:
                device_type = "cpu"
                logger.info("No GPU available, using CPU")
        else:
            # Cloud mode: prioritize CUDA
            if torch.cuda.is_available():
                device_type = "cuda"
                logger.info(
                    "CUDA available",
                    device_count=torch.cuda.device_count(),
                    device_name=torch.cuda.get_device_name(0),
                )
            elif torch.backends.mps.is_available():
                device_type = "mps"
                logger.info("MPS (Metal Performance Shaders) available")
            else:
                device_type = "cpu"
                logger.info("No GPU available, using CPU")
    
    device = torch.device(device_type)
    return device


def get_device_info() -> dict[str, any]:
    """
    Get detailed information about the current device.
    
    Returns:
        Dictionary with device information
    """
    device = get_device()
    info = {
        "device_type": device.type,
        "device_index": device.index,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(device.index or 0),
            "device_capability": torch.cuda.get_device_capability(device.index or 0),
            "memory_allocated_gb": torch.cuda.memory_allocated(device.index or 0) / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved(device.index or 0) / 1024**3,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "chip": "Apple M4 Pro" if "M4" in platform.processor() else platform.processor(),
            "unified_memory_gb": 64,  # M4 Pro spec
            "mlx_available": has_mlx(),
        })
    
    # Add MLX info if available
    if has_mlx():
        try:
            import mlx.core as mx
            info["mlx_version"] = mx.__version__ if hasattr(mx, "__version__") else "available"
            info["mlx_default_device"] = str(mx.default_device())
        except:
            pass
    
    return info


def set_cuda_visible_devices(devices: str | list[int] | None = None) -> None:
    """
    Set CUDA_VISIBLE_DEVICES environment variable.
    
    Args:
        devices: Device indices as string "0,1" or list [0, 1]
    """
    if devices is None:
        devices = get_settings().cuda_visible_devices
    
    if isinstance(devices, list):
        devices = ",".join(map(str, devices))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    logger.info("Set CUDA visible devices", devices=devices)


def get_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """
    Get memory statistics for the device.
    
    Args:
        device: Device to query (defaults to current device)
        
    Returns:
        Dictionary with memory stats in GB
    """
    if device is None:
        device = get_device()
    
    stats = {}
    
    if device.type == "cuda":
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
            "free_gb": (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated(device)
            ) / 1024**3,
            "total_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        }
    elif device.type == "mps":
        # MPS memory stats are limited but we can track allocations
        import psutil
        stats = {
            "system_total_gb": psutil.virtual_memory().total / 1024**3,
            "system_available_gb": psutil.virtual_memory().available / 1024**3,
            "unified_memory_gb": 64,  # M4 Pro spec
            "recommended_vram_gb": 48,  # Conservative estimate for ML workloads
        }
        # Try to get MPS-specific stats if available
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
        except:
            pass
    
    return stats


def empty_cache() -> None:
    """Empty GPU cache for CUDA or MPS."""
    device = get_device()
    
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Emptied CUDA cache")
    elif device.type == "mps" and torch.backends.mps.is_available():
        # MPS cache management
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            logger.info("Emptied MPS cache")
        # Also synchronize to ensure all operations complete
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def get_mlx_device():
    """Get MLX device for Apple Silicon optimized operations."""
    if not has_mlx():
        return None
    
    try:
        import mlx.core as mx
        device = mx.default_device()
        logger.info(f"MLX device available: {device}")
        return device
    except Exception as e:
        logger.warning(f"Failed to get MLX device: {e}")
        return None


def setup_mps_environment():
    """Set optimal environment variables for MPS."""
    mps_env = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.95",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.90",
        "TOKENIZERS_PARALLELISM": "false",
    }
    
    for key, value in mps_env.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
    
    # Don't set default device globally - causes issues with DataLoader
    # Models should be explicitly moved to device


def get_optimal_batch_size(model_type: str = "transformer", device: Optional[torch.device] = None) -> int:
    """Get recommended batch size for the device and model type."""
    if device is None:
        device = get_device()
    
    batch_sizes = {
        "mps": {
            "cnn_small": 64,
            "cnn_large": 32,
            "transformer_tiny": 16,
            "transformer_small": 8,
            "transformer_base": 4,
            "llm_1b": 1,
        },
        "cuda": {
            "cnn_small": 128,
            "cnn_large": 64,
            "transformer_tiny": 32,
            "transformer_small": 16,
            "transformer_base": 8,
            "llm_1b": 4,
        },
        "cpu": {
            "cnn_small": 16,
            "cnn_large": 8,
            "transformer_tiny": 4,
            "transformer_small": 2,
            "transformer_base": 1,
            "llm_1b": 1,
        },
    }
    
    device_type = device.type
    return batch_sizes.get(device_type, batch_sizes["cpu"]).get(model_type, 1)