"""FastAPI application example."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from crown_common import get_logger, get_settings, get_storage

logger = get_logger(__name__)

# Metrics
request_count = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
request_duration = Histogram("http_request_duration_seconds", "HTTP request duration")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    environment: str
    version: str = "0.1.0"


class PredictRequest(BaseModel):
    """Prediction request model."""
    text: str
    max_length: int = 100


class PredictResponse(BaseModel):
    """Prediction response model."""
    prediction: str
    confidence: float
    metadata: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting API service")
    settings = get_settings()
    storage = await get_storage()
    
    # Store in app state
    app.state.settings = settings
    app.state.storage = storage
    
    yield
    
    # Shutdown
    logger.info("Shutting down API service")
    await storage.close()


# Create FastAPI app
app = FastAPI(
    title="Crown Template API",
    description="Template API for Crown services",
    version="0.1.0",
    lifespan=lifespan,
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = app.state.settings
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        environment=settings.environment,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Example prediction endpoint.
    
    In a real service, this would:
    1. Load a model
    2. Process the input
    3. Return predictions
    """
    # Track metrics
    request_count.labels(method="POST", endpoint="/predict").inc()
    
    # Mock prediction logic
    if len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Text too short")
    
    # In real implementation, you'd load model and predict
    prediction = f"Processed: {request.text[:50]}..."
    confidence = 0.95
    
    logger.info(
        "Prediction made",
        text_length=len(request.text),
        confidence=confidence,
    )
    
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        metadata={
            "model": "example-model",
            "version": "0.1.0",
            "max_length": request.max_length,
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Crown Template API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }