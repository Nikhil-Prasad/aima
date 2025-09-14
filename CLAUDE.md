# Claude.md - Project Context

This file provides context for Claude (or other AI assistants) about this project.

## Project Overview

This is a template repository for Crown 621 Python services. It provides a standardized structure for:
- Machine learning experiments
- API services
- Data processing pipelines
- RL agent training

## Key Design Decisions

1. **Local-first development**: Optimized for Mac Mini M-series with MPS support
2. **Cloud GPU burst**: SkyPilot integration for on-demand GPU compute
3. **Storage flexibility**: Local MinIO for dev, S3 for production
4. **Async-first**: All I/O operations use asyncio
5. **Type safety**: Strict mypy checking throughout

## Architecture Patterns

### Storage Layer
- PostgreSQL + pgvector for structured data and embeddings
- Redis for caching and message queuing
- S3/MinIO for blob storage
- All accessed through unified `StorageManager`

### Configuration
- Pydantic Settings for type-safe config
- Environment variables for secrets
- Hydra for complex experiment configs

### GPU Management
- Auto-detection of CUDA/MPS/CPU
- Transparent device handling
- Memory-efficient model loading

## Development Workflow

1. Local development on Mac Mini (64GB RAM)
2. Push code to GitHub
3. Burst to GPU when needed via SkyPilot
4. All experiments tracked in Weights & Biases

## Common Tasks

When working on this codebase:

### Adding a new ML model
1. Create script in `src/scripts/`
2. Use `get_device()` for device selection
3. Implement checkpointing to S3
4. Add W&B tracking

### Adding a new API endpoint
1. Add route to `src/api/main.py`
2. Create Pydantic models for request/response
3. Add Prometheus metrics
4. Write tests in `tests/`

### Running experiments
```bash
# Locally (MPS)
python src/scripts/train_example.py

# On GPU
sky launch skypilot/gpu-a100.yaml
sky ssh crown-gpu
python src/scripts/train_example.py
```

## Testing Requirements

Before committing:
```bash
make format  # Auto-format code
make lint    # Check code style
make type-check  # Verify types
make test    # Run unit tests
```

## Important Files

- `src/crown_common/`: Shared utilities - don't break these!
- `docker/docker-compose.dev.yml`: Local service stack
- `skypilot/*.yaml`: GPU configs - expensive if misconfigured!
- `.env.example`: Document all new env vars here

## Cost Considerations

- A100: ~$1/hour - good for most experiments
- H100: ~$4/hour - only for large models
- Always set auto-shutdown in SkyPilot configs
- Use spot instances when possible

## Getting Help

- Check `README.md` for setup instructions
- Review examples in `src/scripts/`
- Storage patterns in `src/crown_common/storage.py`
- GPU utilities in `src/crown_common/gpu_utils.py`