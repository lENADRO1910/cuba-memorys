"""Hybrid search module: RRF-based scoring with 4 signals.

Scoring formula (Reciprocal Rank Fusion):
    score = 0.35·ts_rank_cd + 0.30·similarity + 0.25·importance + 0.10·freshness

Signals:
1. ts_rank_cd (35%) — Cover Density ranking (positional, better than ts_rank)
2. pg_trgm similarity (30%) — Character-level fuzzy matching
3. Hebbian importance (25%) — Learned relevance via Oja's rule
4. Temporal freshness (10%) — 1/(1 + days_since_access)

References:
- Robertson (2009): Reciprocal Rank Fusion
- ts_rank_cd: PostgreSQL full-text cover density
"""

import hashlib
import time
from typing import Any

# LRU Cache for repeated queries (TTL = 60 seconds)
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
CACHE_TTL: float = 60.0
CACHE_MAX_SIZE: int = 100


def _cache_key(query: str, mode: str, scope: str, limit: int) -> str:
    """Generate cache key from query parameters."""
    raw = f"{query}:{mode}:{scope}:{limit}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(query: str, mode: str, scope: str, limit: int) -> list[dict[str, Any]] | None:
    """Get cached results if still valid.

    Args:
        query: Search query.
        mode: Search mode.
        scope: Search scope.
        limit: Result limit.

    Returns:
        Cached results or None if expired/missing.
    """
    key = _cache_key(query, mode, scope, limit)
    if key in _cache:
        ts, results = _cache[key]
        if time.monotonic() - ts < CACHE_TTL:
            return results
        del _cache[key]
    return None


def cache_set(
    query: str, mode: str, scope: str, limit: int,
    results: list[dict[str, Any]],
) -> None:
    """Store results in cache.

    Args:
        query: Search query.
        mode: Search mode.
        scope: Search scope.
        limit: Result limit.
        results: Results to cache.
    """
    # Evict oldest entries if cache is full
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest_key]

    key = _cache_key(query, mode, scope, limit)
    _cache[key] = (time.monotonic(), results)


def cache_clear() -> None:
    """Clear all cached results."""
    _cache.clear()


# SQL query templates for hybrid search
SEARCH_ENTITIES_SQL = """
SELECT id, name, entity_type, importance, access_count,
    created_at, updated_at,
    (
        0.35 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(name, $1) +
        0.25 * importance +
        0.10 * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0))
    ) AS score
FROM brain_entities
WHERE search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(name, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

SEARCH_OBSERVATIONS_SQL = """
SELECT o.id, o.content, o.observation_type, o.importance,
    o.source, o.created_at, o.last_accessed,
    e.name AS entity_name, e.entity_type,
    (
        0.35 * ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(o.content, $1) +
        0.25 * o.importance +
        0.10 * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0))
    ) AS score
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE o.search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(o.content, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

SEARCH_ERRORS_SQL = """
SELECT id, error_type, error_message, solution, resolved,
    synapse_weight, project, created_at,
    (
        0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(error_message, $1) +
        0.30 * (synapse_weight / GREATEST(
            (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))
    ) AS score
FROM brain_errors
WHERE search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(error_message, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

VERIFY_SQL = """
SELECT
    CASE
        WHEN EXISTS (
            SELECT 1 FROM brain_observations
            WHERE search_vector @@ plainto_tsquery('simple', $1)
              AND similarity(content, $1) > 0.7
        ) THEN 'verified'
        WHEN EXISTS (
            SELECT 1 FROM brain_observations
            WHERE search_vector @@ plainto_tsquery('simple', $1)
              OR similarity(content, $1) > 0.4
        ) THEN 'partial'
        ELSE 'unknown'
    END AS confidence,
    (
        SELECT content FROM brain_observations
        WHERE search_vector @@ plainto_tsquery('simple', $1)
           OR similarity(content, $1) > 0.3
        ORDER BY similarity(content, $1) DESC
        LIMIT 1
    ) AS closest_match
"""
