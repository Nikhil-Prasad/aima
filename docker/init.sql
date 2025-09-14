-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    collection VARCHAR(255) NOT NULL DEFAULT 'default',
    embedding vector(1536),  -- Default to OpenAI embedding size, adjust as needed
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings(collection);
CREATE INDEX IF NOT EXISTS idx_embeddings_metadata ON embeddings USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);

-- Create vector similarity index (IVFFlat)
-- Note: This should be created after you have some data
-- CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_embeddings_updated_at BEFORE UPDATE
    ON embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();