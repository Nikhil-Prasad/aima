"""Storage utilities for PostgreSQL, pgvector, Redis, and S3/MinIO."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import asyncpg
import boto3
import numpy as np
import redis.asyncio as redis
from botocore.client import Config
from pgvector.asyncpg import register_vector
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from crown_common.config import get_settings
from crown_common.logging import get_logger

logger = get_logger(__name__)


class StorageManager:
    """Unified storage manager for all storage backends."""
    
    def __init__(self):
        self.settings = get_settings()
        self._pg_pool: asyncpg.Pool | None = None
        self._redis_client: redis.Redis | None = None
        self._s3_client: Any = None
        self._engine: Any = None
        self._session_factory: sessionmaker | None = None
    
    async def initialize(self) -> None:
        """Initialize all storage connections."""
        await self._init_postgres()
        await self._init_redis()
        self._init_s3()
        logger.info("Storage manager initialized")
    
    async def close(self) -> None:
        """Close all storage connections."""
        if self._pg_pool:
            await self._pg_pool.close()
        if self._redis_client:
            await self._redis_client.close()
        if self._engine:
            await self._engine.dispose()
        logger.info("Storage manager closed")
    
    # PostgreSQL + pgvector
    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection pool with pgvector support."""
        # Raw asyncpg pool for direct queries
        self._pg_pool = await asyncpg.create_pool(
            str(self.settings.database_url),
            min_size=5,
            max_size=20,
            command_timeout=60,
            init=self._init_pgvector_connection,
        )
        
        # SQLAlchemy async engine for ORM
        self._engine = create_async_engine(
            str(self.settings.database_url),
            echo=self.settings.is_development,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    @staticmethod
    async def _init_pgvector_connection(conn: asyncpg.Connection) -> None:
        """Initialize pgvector for a connection."""
        await register_vector(conn)
    
    @asynccontextmanager
    async def get_db_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a database connection from the pool."""
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL not initialized")
        
        async with self._pg_pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a SQLAlchemy async session."""
        if not self._session_factory:
            raise RuntimeError("SQLAlchemy not initialized")
        
        async with self._session_factory() as session:
            yield session
    
    async def store_embedding(
        self,
        embedding: np.ndarray,
        metadata: dict[str, Any],
        collection: str = "default",
    ) -> str:
        """
        Store an embedding with metadata in pgvector.
        
        Args:
            embedding: Numpy array of the embedding
            metadata: Associated metadata
            collection: Collection name for organizing embeddings
            
        Returns:
            ID of the stored embedding
        """
        async with self.get_db_connection() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO embeddings (collection, embedding, metadata)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                collection,
                embedding.tolist(),
                metadata,
            )
            return str(result)
    
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        collection: str = "default",
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search in pgvector.
        
        Args:
            query_embedding: Query embedding vector
            collection: Collection to search in
            limit: Maximum number of results
            threshold: Optional similarity threshold
            
        Returns:
            List of similar items with metadata and scores
        """
        async with self.get_db_connection() as conn:
            query = """
                SELECT id, metadata, embedding <=> $1 as distance
                FROM embeddings
                WHERE collection = $2
            """
            params = [query_embedding.tolist(), collection]
            
            if threshold is not None:
                query += " AND embedding <=> $1 <= $3"
                params.append(threshold)
            
            query += " ORDER BY embedding <=> $1 LIMIT $3"
            params.append(limit)
            
            results = await conn.fetch(query, *params)
            
            return [
                {
                    "id": str(row["id"]),
                    "metadata": row["metadata"],
                    "score": 1 - row["distance"],  # Convert distance to similarity
                }
                for row in results
            ]
    
    # Redis
    async def _init_redis(self) -> None:
        """Initialize Redis connection."""
        self._redis_client = redis.from_url(
            str(self.settings.redis_url),
            decode_responses=True,
        )
        await self._redis_client.ping()
    
    @property
    def redis(self) -> redis.Redis:
        """Get Redis client."""
        if not self._redis_client:
            raise RuntimeError("Redis not initialized")
        return self._redis_client
    
    # S3/MinIO
    def _init_s3(self) -> None:
        """Initialize S3/MinIO client."""
        if self.settings.is_production and self.settings.aws_access_key_id:
            # Production: Use AWS S3
            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_default_region,
            )
            self._bucket = self.settings.s3_bucket
        else:
            # Development: Use MinIO
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=f"http://{self.settings.minio_endpoint}",
                aws_access_key_id=self.settings.minio_access_key,
                aws_secret_access_key=self.settings.minio_secret_key,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            self._bucket = self.settings.minio_bucket
            
            # Create bucket if it doesn't exist
            try:
                self._s3_client.head_bucket(Bucket=self._bucket)
            except:
                self._s3_client.create_bucket(Bucket=self._bucket)
    
    @property
    def s3(self) -> Any:
        """Get S3 client."""
        if not self._s3_client:
            raise RuntimeError("S3 not initialized")
        return self._s3_client
    
    @property
    def bucket(self) -> str:
        """Get current S3 bucket name."""
        return self._bucket


# Global storage instance
_storage: StorageManager | None = None


async def get_storage() -> StorageManager:
    """Get or create the global storage manager."""
    global _storage
    
    if _storage is None:
        _storage = StorageManager()
        await _storage.initialize()
    
    return _storage