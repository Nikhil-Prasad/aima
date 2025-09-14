"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client: TestClient):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "crown-template"


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert data["message"] == "Crown Template API"


def test_predict_endpoint(client: TestClient):
    """Test prediction endpoint."""
    # Valid request
    response = client.post(
        "/predict",
        json={"text": "This is a test input for prediction", "max_length": 50},
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "metadata" in data
    
    # Invalid request (text too short)
    response = client.post(
        "/predict",
        json={"text": "short"},
    )
    assert response.status_code == 400


def test_metrics_endpoint(client: TestClient):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text