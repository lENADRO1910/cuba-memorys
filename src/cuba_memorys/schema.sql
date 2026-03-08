-- Cuba-Memorys: PostgreSQL Schema
-- Extensions: pg_trgm (fuzzy), tsvector (full-text). Both built-in since PG 9.1+.
-- Minimum: PostgreSQL 12 (generated columns).

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Entities (knowledge graph nodes)
CREATE TABLE IF NOT EXISTS brain_entities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    importance FLOAT DEFAULT 0.5
        CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(entity_type, ''))
    ) STORED
);

-- 2. Observations (facts, decisions, lessons attached to entities)
CREATE TABLE IF NOT EXISTS brain_observations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    observation_type TEXT DEFAULT 'fact'
        CHECK (observation_type IN (
            'fact', 'decision', 'lesson', 'preference',
            'error', 'solution', 'context', 'tool_usage'
        )),
    importance FLOAT DEFAULT 0.5
        CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    source TEXT DEFAULT 'agent'
        CHECK (source IN ('agent', 'error_detection', 'user', 'consolidation', 'inference')),
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED
);

-- 3. Relations (knowledge graph edges)
CREATE TABLE IF NOT EXISTS brain_relations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    from_entity UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    to_entity UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0
        CHECK (strength >= 0.0 AND strength <= 1.0),
    bidirectional BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(from_entity, to_entity, relation_type)
);

-- 4. Error Memory
CREATE TABLE IF NOT EXISTS brain_errors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    project TEXT DEFAULT 'default',
    solution TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    attempts INT DEFAULT 0,
    synapse_weight FLOAT DEFAULT 1.0
        CHECK (synapse_weight >= 0.0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple',
            coalesce(error_type, '') || ' ' ||
            coalesce(error_message, '') || ' ' ||
            coalesce(solution, ''))
    ) STORED
);

-- 5. Sessions
CREATE TABLE IF NOT EXISTS brain_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_name TEXT,
    goals JSONB DEFAULT '[]',
    summary TEXT,
    outcome TEXT CHECK (outcome IN ('success', 'partial', 'failed', 'abandoned', NULL)),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_search ON brain_entities USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_entities_trgm ON brain_entities USING GIN(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_type ON brain_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_obs_search ON brain_observations USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_obs_trgm ON brain_observations USING GIN(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON brain_observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_obs_type ON brain_observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_obs_importance ON brain_observations(importance DESC);
CREATE INDEX IF NOT EXISTS idx_errors_search ON brain_errors USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_errors_trgm ON brain_errors USING GIN(error_message gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_errors_project ON brain_errors(project);
CREATE INDEX IF NOT EXISTS idx_errors_resolved ON brain_errors(resolved);
CREATE INDEX IF NOT EXISTS idx_relations_from ON brain_relations(from_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to ON brain_relations(to_entity);
