"""Example training script demonstrating the template capabilities."""

import asyncio
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from crown_common import get_device, get_logger, get_settings, get_storage

logger = get_logger(__name__)


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


async def save_checkpoint_to_s3(
    model: nn.Module,
    epoch: int,
    storage: "StorageManager",
    key: str,
) -> None:
    """Save model checkpoint to S3/MinIO."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "timestamp": str(Path.cwd()),
    }
    
    # Save locally first
    local_path = f"/tmp/checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, local_path)
    
    # Upload to S3
    storage.s3.upload_file(local_path, storage.bucket, key)
    logger.info("Saved checkpoint to S3", key=key, epoch=epoch)
    
    # Clean up local file
    Path(local_path).unlink()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
        # Log to W&B
        if batch_idx % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": epoch * len(dataloader) + batch_idx,
            })
    
    return total_loss / len(dataloader)


async def main():
    """Main training function."""
    # Get configuration
    settings = get_settings()
    device = get_device()
    
    logger.info(
        "Starting training",
        device=str(device),
        service=settings.service_name,
        environment=settings.environment,
    )
    
    # Initialize W&B
    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        mode=settings.wandb_mode,
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
            "hidden_size": 64,
        },
    )
    
    # Initialize storage
    storage = await get_storage()
    
    # Create synthetic dataset for demo
    n_samples = 1000
    input_size = 10
    X = torch.randn(n_samples, input_size)
    y = torch.randn(n_samples)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Log model info
    logger.info(
        "Model initialized",
        parameters=sum(p.numel() for p in model.parameters()),
        device=str(device),
    )
    
    # Training loop
    for epoch in range(10):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
        
        logger.info("Epoch completed", epoch=epoch, avg_loss=avg_loss)
        wandb.log({"epoch/loss": avg_loss, "epoch": epoch})
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_key = f"checkpoints/{wandb.run.id}/epoch_{epoch}.pt"
            await save_checkpoint_to_s3(model, epoch, storage, checkpoint_key)
    
    # Final save
    final_key = f"models/{wandb.run.id}/final_model.pt"
    await save_checkpoint_to_s3(model, 10, storage, final_key)
    
    wandb.finish()
    logger.info("Training completed")


if __name__ == "__main__":
    asyncio.run(main())