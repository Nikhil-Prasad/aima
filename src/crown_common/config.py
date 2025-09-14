"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )
    
    # Service
    service_name: str = Field(default="crown-template", description="Service name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment"
    )
    
    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://crown:crown@localhost:5432/crown_dev",
        description="PostgreSQL connection URL",
    )
    
    # Redis
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    
    # Storage
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_bucket: str = Field(default="crown-artifacts", description="MinIO bucket")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO")
    
    # AWS (production)
    aws_access_key_id: str | None = Field(default=None, description="AWS access key")
    aws_secret_access_key: str | None = Field(default=None, description="AWS secret key")
    aws_default_region: str = Field(default="us-east-1", description="AWS region")
    s3_bucket: str | None = Field(default=None, description="S3 bucket name")
    
    # Weights & Biases
    wandb_project: str = Field(default="crown-experiments", description="W&B project")
    wandb_entity: str | None = Field(default=None, description="W&B entity")
    wandb_mode: Literal["offline", "online", "disabled"] = Field(
        default="offline", description="W&B mode"
    )
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log format"
    )
    
    # Compute Mode
    compute_mode: Literal["local", "cloud", "auto"] = Field(
        default="local", description="Compute mode: local (MPS/MLX priority) or cloud (CUDA priority)"
    )
    
    # GPU
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto", description="Device selection"
    )
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    set_default_device: bool = Field(
        default=True, description="Set torch default device (recommended for MPS)"
    )
    
    # MPS Settings (M4 Pro optimized)
    mps_fallback: bool = Field(default=True, description="Enable MPS fallback for unsupported ops")
    mps_high_watermark: float = Field(default=0.95, description="MPS memory high watermark")
    mps_low_watermark: float = Field(default=0.90, description="MPS memory low watermark")
    
    # Training
    mixed_precision: Literal["no", "fp16", "bf16", "fp16_mps"] = Field(
        default="fp16_mps", description="Mixed precision training mode"
    )
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    dataloader_num_workers: int = Field(default=-1, description="DataLoader workers (-1 for auto)")
    seed: int = Field(default=621, description="Random seed for reproducibility")
    
    # Model
    model_name: str = Field(
        default="mistralai/Mistral-7B-v0.1", description="Default model name"
    )
    model_cache_dir: str = Field(default="./models", description="Model cache directory")
    
    # Sentry
    sentry_dsn: str | None = Field(default=None, description="Sentry DSN")
    
    # Feature flags
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode (MPS/MLX priority)."""
        return self.compute_mode == "local"
    
    @property
    def is_cloud_mode(self) -> bool:
        """Check if running in cloud mode (CUDA priority)."""
        return self.compute_mode == "cloud"
    
    @property
    def optimal_num_workers(self) -> int:
        """Get optimal number of dataloader workers."""
        if self.dataloader_num_workers >= 0:
            return self.dataloader_num_workers
        import os
        # For M4 Pro, use CPU count - 1 for optimal performance
        return max(0, os.cpu_count() - 1) if os.cpu_count() else 0
    
    @property
    def storage_backend(self) -> Literal["minio", "s3"]:
        """Determine storage backend based on mode."""
        if self.is_local_mode or not self.s3_bucket:
            return "minio"
        return "s3"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()