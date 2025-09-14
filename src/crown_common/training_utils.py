"""Training utilities optimized for MPS and MLX."""

import contextlib
import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from crown_common.config import get_settings
from crown_common.gpu_utils import get_device, empty_cache, setup_mps_environment
from crown_common.logging import get_logger

logger = get_logger(__name__)


def setup_training_environment():
    """Setup optimal training environment for M4 Pro."""
    settings = get_settings()
    
    # Set MPS environment variables
    if settings.is_local_mode:
        setup_mps_environment()
    
    # Set seed for reproducibility
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    
    # Set deterministic operations if needed (may impact performance)
    # torch.use_deterministic_algorithms(True)
    
    logger.info(f"Training environment setup complete, seed={settings.seed}")


def get_autocast_context(device: Optional[torch.device] = None):
    """Get appropriate autocast context for the device."""
    if device is None:
        device = get_device()
    
    settings = get_settings()
    
    # MPS doesn't support autocast in PyTorch 2.2
    # Just return no-op context for MPS
    if device.type == "mps":
        return contextlib.nullcontext()
    elif device.type == "cuda":
        if settings.mixed_precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        elif settings.mixed_precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    
    # No autocast for CPU or when disabled
    return contextlib.nullcontext()


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 0.0005,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """Create optimizer with recommended settings."""
    if optimizer_type.lower() == "adamw":
        # AdamW with decoupled weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={lr}")
    return optimizer


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs
) -> DataLoader:
    """Create DataLoader with MPS-optimized settings."""
    settings = get_settings()
    
    # Determine number of workers
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = settings.optimal_num_workers
    
    # MPS-specific optimizations
    device = get_device()
    if device.type == "mps":
        # MPS works best with persistent workers
        persistent_workers = num_workers > 0
        # Don't pin memory for MPS
        pin_memory = False
        # Moderate prefetch factor
        prefetch_factor = kwargs.get("prefetch_factor", 2)
    else:
        persistent_workers = kwargs.get("persistent_workers", num_workers > 0)
        pin_memory = kwargs.get("pin_memory", device.type == "cuda")
        prefetch_factor = kwargs.get("prefetch_factor", 2)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    logger.info(
        f"Created DataLoader: batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )
    
    return loader


class GradientAccumulator:
    """Helper for gradient accumulation with mixed precision."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        
        # GradScaler is not reliable on MPS
        device = get_device()
        settings = get_settings()
        self.use_grad_scaler = (
            device.type == "cuda" and settings.mixed_precision in ["fp16", "bf16"]
        )
        
        if self.use_grad_scaler:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with optional scaling."""
        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.step_count += 1
    
    def step(self) -> bool:
        """Optimizer step if accumulation is complete."""
        if self.step_count % self.accumulation_steps != 0:
            return False
        
        # Gradient clipping
        if self.max_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        return True


class MemoryManager:
    """Memory management for MPS and CUDA."""
    
    def __init__(self, empty_cache_freq: int = 100):
        self.empty_cache_freq = empty_cache_freq
        self.step_count = 0
        self.device = get_device()
    
    def step(self):
        """Periodic cache clearing."""
        self.step_count += 1
        if self.step_count % self.empty_cache_freq == 0:
            empty_cache()
            logger.debug(f"Emptied cache at step {self.step_count}")
    
    def reset(self):
        """Reset counter and clear cache."""
        self.step_count = 0
        empty_cache()


def model_to_channels_last(model: nn.Module) -> nn.Module:
    """Convert CNN model to channels_last format for better performance."""
    device = get_device()
    
    # Channels last is beneficial for CNNs on MPS and CUDA
    if device.type in ["mps", "cuda"]:
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Converted model to channels_last format")
        except Exception as e:
            logger.warning(f"Could not convert to channels_last: {e}")
    
    return model


def get_mlx_model(model_name: str, **kwargs):
    """Load a model using MLX if available."""
    try:
        import mlx.core as mx
        import mlx.nn as mnn
        from mlx.utils import tree_map
        
        # MLX model loading logic here
        logger.info(f"Loading MLX model: {model_name}")
        # This is a placeholder - actual implementation depends on model type
        return None
    except ImportError:
        logger.warning("MLX not available, falling back to PyTorch")
        return None


def train_step_with_logging(
    model: nn.Module,
    batch: tuple,
    optimizer: Optimizer,
    criterion: nn.Module,
    accumulator: Optional[GradientAccumulator] = None,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Single training step with automatic mixed precision."""
    if device is None:
        device = get_device()
    
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Forward pass with autocast
    with get_autocast_context(device):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward pass
    if accumulator:
        accumulator.backward(loss)
        step_performed = accumulator.step()
    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_performed = True
    
    # Calculate metrics
    metrics = {
        "loss": loss.item(),
        "step_performed": step_performed,
    }
    
    # Add accuracy for classification
    if outputs.dim() > 1 and targets.dim() == 1:
        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).float().mean().item()
        metrics["accuracy"] = accuracy
    
    return metrics