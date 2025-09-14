"""Tests for storage utilities."""

import asyncio

import numpy as np
import pytest

from crown_common.storage import StorageManager


@pytest.fixture
async def storage():
    """Create storage manager for tests."""
    manager = StorageManager()
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.mark.asyncio
async def test_redis_connection(storage: StorageManager):
    """Test Redis connectivity."""
    # Set and get a value
    await storage.redis.set("test_key", "test_value")
    value = await storage.redis.get("test_key")
    assert value == "test_value"
    
    # Clean up
    await storage.redis.delete("test_key")


@pytest.mark.asyncio
async def test_s3_operations(storage: StorageManager):
    """Test S3/MinIO operations."""
    # Create test data
    test_key = "test/example.txt"
    test_content = b"Hello, Crown!"
    
    # Upload
    storage.s3.put_object(
        Bucket=storage.bucket,
        Key=test_key,
        Body=test_content,
    )
    
    # Download
    response = storage.s3.get_object(
        Bucket=storage.bucket,
        Key=test_key,
    )
    downloaded = response["Body"].read()
    
    assert downloaded == test_content
    
    # Clean up
    storage.s3.delete_object(
        Bucket=storage.bucket,
        Key=test_key,
    )


@pytest.mark.asyncio
async def test_pgvector_operations(storage: StorageManager):
    """Test PostgreSQL with pgvector."""
    # Create test embedding
    embedding = np.random.randn(1536).astype(np.float32)
    metadata = {"test": True, "name": "test_embedding"}
    
    # Store embedding
    embedding_id = await storage.store_embedding(
        embedding=embedding,
        metadata=metadata,
        collection="test",
    )
    
    assert embedding_id is not None
    
    # Search for similar embeddings
    results = await storage.similarity_search(
        query_embedding=embedding,
        collection="test",
        limit=5,
    )
    
    assert len(results) > 0
    assert results[0]["id"] == embedding_id
    assert results[0]["score"] > 0.99  # Should be nearly identical