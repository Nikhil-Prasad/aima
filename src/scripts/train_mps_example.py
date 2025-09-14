#!/usr/bin/env python
"""Example training script optimized for M4 Pro with MPS."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from crown_common.config import get_settings
from crown_common.gpu_utils import (
    get_device,
    get_device_info,
    setup_mps_environment,
    empty_cache,
    get_optimal_batch_size,
)
from crown_common.logging import get_logger
from crown_common.training_utils import (
    setup_training_environment,
    create_optimizer,
    create_dataloader,
    GradientAccumulator,
    MemoryManager,
    model_to_channels_last,
    train_step_with_logging,
    get_autocast_context,
)

logger = get_logger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_data_loaders(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 data loaders."""
    settings = get_settings()
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create dataloaders with MPS-optimized settings
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for evaluation
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader


def validate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validation loop."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Use autocast for validation too
            with get_autocast_context(device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        "val_loss": test_loss / len(test_loader),
        "val_accuracy": 100.0 * correct / total,
    }


def train(args):
    """Main training loop."""
    # Setup environment
    setup_training_environment()
    settings = get_settings()
    
    # Get device
    device = get_device()
    logger.info(f"Device info: {get_device_info()}")
    
    # Initialize wandb
    if settings.wandb_mode != "disabled":
        wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "device": device.type,
                "mixed_precision": settings.mixed_precision,
            },
        )
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Optimize for channels_last memory format (beneficial for CNNs)
    if device.type in ["mps", "cuda"]:
        model = model_to_channels_last(model)
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type="adamw",
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    
    # Gradient accumulator for mixed precision
    accumulator = GradientAccumulator(
        model,
        optimizer,
        accumulation_steps=settings.gradient_accumulation_steps,
        max_grad_norm=1.0,
    )
    
    # Memory manager
    memory_manager = MemoryManager(empty_cache_freq=100)
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Training step with automatic mixed precision
            metrics = train_step_with_logging(
                model,
                (inputs, targets),
                optimizer,
                criterion,
                accumulator,
                device,
            )
            
            train_loss += metrics["loss"]
            
            # Calculate accuracy
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": train_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                })
            
            # Memory management
            memory_manager.step()
            
            # Log to wandb
            if settings.wandb_mode != "disabled" and metrics["step_performed"]:
                wandb.log({
                    "train/loss": metrics["loss"],
                    "train/accuracy": 100.0 * correct / total,
                })
        
        # Validation
        val_metrics = validate(model, test_loader, criterion, device)
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train Acc: {100.0*correct/total:.2f}%, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
        )
        
        if settings.wandb_mode != "disabled":
            wandb.log({
                "epoch": epoch + 1,
                "val/loss": val_metrics["val_loss"],
                "val/accuracy": val_metrics["val_accuracy"],
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss / len(train_loader),
                "val_metrics": val_metrics,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clear cache at end of epoch
        memory_manager.reset()
    
    if settings.wandb_mode != "disabled":
        wandb.finish()
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="MPS-optimized training example")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64 for M4 Pro)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save-freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Auto-adjust batch size if not specified
    if args.batch_size == 64:
        device = get_device()
        args.batch_size = get_optimal_batch_size("cnn_small", device)
        logger.info(f"Auto-selected batch size: {args.batch_size}")
    
    train(args)


if __name__ == "__main__":
    main()