# Crown 621 Architecture Memory

> Created: 2025-08-05  
> Source: architecture.md chat transcript analysis

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Cloud Economics](#2-cloud-economics)
3. [Use Cases](#3-use-cases)
4. [Stack Decisions](#4-stack-decisions)
5. [Python Workflow](#5-python-workflow)
6. [Storage Primitives](#6-storage-primitives)
7. [Experiment Tracking](#7-experiment-tracking)
8. [Reinforcement Learning](#8-reinforcement-learning)
9. [GPU Workflow](#9-gpu-workflow)
10. [Template Structure](#10-template-structure)
11. [AWS Setup](#11-aws-setup)
12. [Development Workflow](#12-development-workflow)

---

## 1. Hardware Requirements

**Summary**: GPU memory requirements for running various LLM sizes locally, from 7B to 70B+ models

### Key Decisions
- ≤ 13B models need ~26GB VRAM (1x RTX 4090)
- 70B models need 40-48GB in 4-bit or ~140GB in FP16
- Hardware tiers from $3.5k prosumer to $35k+ accelerator-grade
- **Conclusion: Cloud is more economical for most use cases**

### Details
Model size requirements:
- ≤ 13B FP16 needs ~26GB VRAM (1x RTX 4090)
- 30-34B in 4-8 bit needs 24-32GB
- 70B 4-bit GPTQ needs 40-48GB
- 70B FP16 needs ~140GB

Hardware tiers:
- A) Prosumer $3.5-4k for 7-34B models
- B) Enthusiast multi-GPU $7-8k for 70B
- C) Workstation-class $15-20k with NVLink for FP16 70B
- D) Accelerator-grade $35k+ for 100B+ models

---

## 2. Cloud Economics

**Summary**: Cost analysis showing cloud GPUs are cheaper unless running >17 hours/day

### Key Decisions
- Cloud 4090 at $0.34/hr beats local if <17hr/day over 2 years
- H100 only available via cloud for prosumers
- Spot instances on Vast.ai/RunPod much cheaper than AWS
- **Decision: Use cloud for all GPU workloads**

### Details
- Local 4090 workstation costs ~$3,500 + $0.07/hr power
- Cloud 4090 median $0.34/hr spot
- Break-even at 17hr/day for 2 years
- Cloud RTX 6000 Ada $0.77/hr, H100 $6.75/hr on AWS

Practical workflows:
- Occasional inference via RunPod/Vast.ai
- Interactive tinkering on Mac mini with 7-13B models
- Long jobs on spot GPUs with checkpointing

---

## 3. Use Cases

**Summary**: Three main research tracks: personal LLM, DAO agents, and large model integration

### Key Decisions
- Personal model: 7-13B with LoRA + RAG for memory
- DAO agents: 3-7B with RL training
- Large models: API-based for 400B+ (DeepSeek/Llama)
- 20B models for specialized info gathering

### Research Tracks
1. **Personal autobiographical model**
   - 7-13B (Mistral/Gemma) with LoRA fine-tuning
   - External memory via RAG/KG
   
2. **Small RL agents for DAO Core**
   - 3-7B models using PPO/CleanRL
   
3. **Large models for Ton618 planner**
   - DeepSeek-600B/Llama-405B via API
   
4. **Domain-specific optimization**
   - 20B models for information gathering

---

## 4. Stack Decisions

**Summary**: Core stack decisions prioritizing Python expertise and pragmatic choices

### Key Decisions
- Python 3.12 for most services (intermediate/advanced skill)
- Rust only for performance-critical components
- Docker + docker-compose for all services
- Postgres + pgvector as primary datastore
- SkyPilot for GPU orchestration

### Stack Details
- **Runtime**: Python 3.12 + poetry monorepo
- **System-level**: Rust 1.78 only for verifiable-compute wedge and HNSW ANN
- **Container**: Docker everywhere
- **Orchestration**: Nomad-lite on EC2
- **Data**: Postgres 16 + pgvector, S3/MinIO for blobs
- **Inference**: Spot H100/A100 via SkyPilot
- **Observability**: Prometheus + Grafana, Loki logs

---

## 5. Python Workflow

**Summary**: Standardized Python development practices across all repos

### Key Decisions
- Template repo pattern (not monorepo)
- Ruff + Mypy + pytest for code quality
- FastAPI for APIs, Redis for queuing
- Apache Burr for state management (not LangGraph)
- DTOs with OpenAI JSON mode

### Standards
- Each service inherits from `template-python-svc`
- Pre-commit: Ruff → Mypy → Pytest
- Runtime: FastAPI + Uvicorn
- Async: asyncio with TaskGroup
- State: Apache Burr graph executor
- LLM integration: Pydantic DTOs + OpenAI JSON mode
- Dependencies: uv (fast Rust resolver) replacing Poetry for services

---

## 6. Storage Primitives

**Summary**: Unified storage primitives for all experiments

### Key Decisions
- Postgres 16 + pgvector for relational + vectors
- MinIO locally → S3 in production
- Redis 7 for queuing and caching
- Qdrant optional for >10M vectors

### Implementation
- Every repo gets `docker-compose.dev.yml` with:
  - Postgres+pgvector (ankane/pgvector:16)
  - MinIO (S3-compatible)
  - Redis 7
  - Optional Qdrant
- `crown_common.storage` provides unified interface
- Local dev uses localhost endpoints
- Prod switches via ENV vars to RDS/S3
- pgvector handles up to ~10M embeddings with HNSW index

---

## 7. Experiment Tracking

**Summary**: Standardized logging, metrics, and ML experiment tracking

### Key Decisions
- Weights & Biases for ML experiments
- structlog → Loki for structured logging
- Prometheus + Grafana for metrics
- Offline mode for private experiments

### Details
- W&B integration with offline→sync mode for air-gapped training
- structlog for JSON logging to Loki
- Prometheus FastAPI middleware for metrics
- Sentry for error capture
- dotenv-vault locally, AWS Secrets Manager in prod
- All experiments log to same W&B project
- Artifacts to S3/MinIO

---

## 8. Reinforcement Learning

**Summary**: Standardized RL training infrastructure

### Key Decisions
- Gymnasium for environment API
- CleanRL + HF TRL for training
- Redis Streams for experience buffer
- Hydra for config management

### Template Structure
- `envs/` folder with Gymnasium environments
- `scripts/train_ppo.py` using CleanRL
- `train_ppo_trl.py` for LLM RLHF
- AsyncVectorEnv for parallel episodes
- Hydra YAML configs
- W&B for tracking reward curves
- Optional Ray RLlib for distributed training

---

## 9. GPU Workflow

**Summary**: Local CPU/MPS development with cloud GPU burst

### Key Decisions
- VS Code dev containers for consistency
- SkyPilot for GPU provisioning
- Auto device selection (MPS/CUDA/CPU)
- Spot instances with idle shutdown

### Workflow
1. Local Mac (MPS) development
2. `git push`
3. SkyPilot launches spot GPU
4. VS Code Remote-SSH
5. `git pull` on remote
6. Run training
7. Auto-shutdown

### Implementation
- `get_device()` helper auto-selects MPS locally, CUDA on EC2
- Dev containers have CPU/MPS Dockerfile and GPU Dockerfile.gpu
- SkyPilot YAMLs for A10/A100/H100 with autostop

---

## 10. Template Structure

**Summary**: Cookiecutter-style template for all Python services

### Key Decisions
- Single template repo, not monorepo
- Each experiment is separate repo from template
- Shared code via crown-common package
- Public/private repo split for resume building

### Structure
```
template-python-svc/
├── .devcontainer/
├── docker-compose.dev.yml
├── skypilot/
├── src/
│   ├── crown_common/
│   ├── envs/
│   └── scripts/
├── pyproject.toml
├── uv.lock
├── Makefile
└── .env.example
```

### Usage
```bash
gh repo create crown-X --template template-python-svc
```

---

## 11. AWS Setup

**Summary**: AWS configuration needed before using template

### Key Decisions
- Dedicated Crown Dev account/OU
- IAM user for SkyPilot with EC2/S3/CloudWatch access
- S3 bucket for artifacts
- CloudWatch Lambda for idle shutdown

### One-Time Setup
1. Crown Dev AWS account with budget alerts
2. IAM user 'skypilot-dev' with programmatic access
3. EC2 key pair for VS Code SSH
4. VPC/Security Group (or use SSM)
5. S3 bucket `crown-artifacts`
6. Service quotas for GPU instances
7. CloudWatch Lambda for 30min idle shutdown

---

## 12. Development Workflow

**Summary**: Complete workflow from local dev to GPU training

### Key Decisions
- Push-pull-run cycle
- Local dev on Mac, GPU burst on AWS
- Git-based deployment
- Cost optimization via spot + autoshutdown

### Step-by-Step Workflow
1. Local changes on Mac
2. `git push`
3. `sky launch gpu.yaml`
4. VS Code Remote-SSH to instance
5. `git pull` in remote
6. Run training (logs to W&B)
7. Checkpoints to S3
8. `sky down` or auto-shutdown

### Costs
- A100 spot: ~$0.70-1/hr
- H100: ~$4-5/hr
- 20hr/mo = $14-100

---

## Quick Reference

### Most Important Commands
```bash
# Create new service
gh repo create crown-X --template template-python-svc

# Launch GPU
sky launch skypilot/gpu-A100.yaml

# Connect VS Code
code --folder-uri "vscode-remote://ssh-remote+crown-gpu/home/ubuntu/crown-X"

# Shutdown
sky down crown-gpu
```

### Key Files to Remember
- `docker-compose.dev.yml` - Local services
- `skypilot/*.yaml` - GPU configs
- `src/crown_common/` - Shared utilities
- `.devcontainer/` - VS Code setup
- `pyproject.toml` - Dependencies

### Cost Optimization Tips
- Always use spot instances
- Set autostop in SkyPilot YAMLs
- Use CloudWatch Lambda for idle shutdown
- Develop locally (MPS), train on cloud (CUDA)
