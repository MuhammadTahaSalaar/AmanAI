-- AmanAI Supabase schema
-- Run once against your Supabase project (SQL editor or psql).

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content      TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}',
    embedding    vector(1024),
    fts          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    content_hash TEXT UNIQUE,
    source_kind  TEXT DEFAULT 'seed',
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest-neighbour search (cosine distance)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS documents_fts_gin
    ON documents USING gin (fts);

-- ---------------------------------------------------------------------------
-- Hybrid retrieval RPC (vector cosine + BM25-style FTS fused via RRF)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    query_text      text,
    match_count     int,
    rrf_k           int DEFAULT 60
)
RETURNS TABLE(
    id           bigint,
    content      text,
    metadata     jsonb,
    score        float
)
LANGUAGE plpgsql
AS $$
DECLARE
    _use_fts boolean;
BEGIN
    -- Only use FTS when query_text is non-empty after trimming
    _use_fts := (query_text IS NOT NULL AND length(trim(query_text)) > 0);

    RETURN QUERY
    WITH vector_ranked AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding) AS rank
        FROM documents d
        WHERE d.embedding IS NOT NULL
    ),
    fts_ranked AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(d.fts, websearch_to_tsquery('english', query_text)) DESC
            ) AS rank
        FROM documents d
        WHERE _use_fts AND d.fts @@ websearch_to_tsquery('english', query_text)
    ),
    rrf AS (
        SELECT
            COALESCE(vr.id, fr.id) AS id,
            COALESCE(1.0 / (rrf_k + vr.rank), 0.0)
                + COALESCE(1.0 / (rrf_k + fr.rank), 0.0) AS rrf_score
        FROM vector_ranked vr
        FULL OUTER JOIN fts_ranked fr ON vr.id = fr.id
    )
    SELECT
        d.id,
        d.content,
        d.metadata,
        rrf.rrf_score::float AS score
    FROM rrf
    JOIN documents d ON d.id = rrf.id
    ORDER BY rrf.rrf_score DESC
    LIMIT match_count;
END;
$$;
