import re
import threading
import time
from typing import Any

RRF_K: int = 60  # Cormack (2009) standard constant
DEDUP_OVERLAP: float = 0.75  # V2: word-overlap threshold for semantic dedup


def _text_overlap(a: str, b: str) -> float:
    """Word-overlap ratio between two texts.

    Jaccard-like but uses min(len) as denominator to catch
    subset matches (e.g. entity name inside observation content).

    Returns:
        Overlap ratio in [0, 1].
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    return intersection / min(len(words_a), len(words_b))


def rrf_fuse(
    signal_rankings: list[list[dict[str, Any]]],
    dedup_threshold: float = DEDUP_OVERLAP,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across N ranked signal lists.

    V2: Post-fusion dedup removes semantic duplicates that appear
    across entity and observation signals (Cormack 2009 + dedup).

    Args:
        signal_rankings: List of ranked result lists from different signals.
        dedup_threshold: Word-overlap threshold for dedup (default 0.75).

    Returns:
        Fused, deduplicated results sorted by RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    for ranking in signal_rankings:
        for rank, item in enumerate(ranking):
            key = str(item.get("id", item.get("name", "")))
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            if key not in items:
                items[key] = item
    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    fused = [
        {**items[k], "rrf_score": round(scores[k], 4)} for k in sorted_keys
    ]
    # V2: Remove semantic duplicates from fused results
    unique: list[dict[str, Any]] = []
    for item in fused:
        content = item.get("content", item.get("name", ""))
        if not any(
            _text_overlap(content, u.get("content", u.get("name", "")))
            > dedup_threshold
            for u in unique
        ):
            unique.append(item)
    return unique

_cache: dict[int, tuple[float, list[dict[str, Any]]]] = {}
_cache_lock = threading.Lock()
CACHE_TTL: float = 60.0
CACHE_MAX_SIZE: int = 100

_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bno\b|\bnot\b|\bnever\b|\bya no\b|\bno es\b", re.IGNORECASE),
    re.compile(r"\bcambió\s+de\b|\bchanged\s+from\b|\breplaced\b", re.IGNORECASE),
    re.compile(r"\ben vez de\b|\binstead of\b|\brather than\b", re.IGNORECASE),
    re.compile(r"\bremoved\b|\belimina\b|\bdeprecated\b|\bobsolete\b", re.IGNORECASE),
    re.compile(r"\bwas\b.*\bnow\b|\bantes\b.*\bahora\b", re.IGNORECASE),
]

def _cache_key(query: str, mode: str, scope: str, limit: int) -> int:
    return hash((query, mode, scope, limit))

def cache_get(query: str, mode: str, scope: str, limit: int) -> list[dict[str, Any]] | None:
    key = _cache_key(query, mode, scope, limit)
    with _cache_lock:
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
    key = _cache_key(query, mode, scope, limit)
    with _cache_lock:
        if len(_cache) >= CACHE_MAX_SIZE:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest_key]
        _cache[key] = (time.monotonic(), results)

def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()

def has_negation(text_a: str, text_b: str) -> bool:
    combined = f"{text_a} {text_b}"
    return any(pattern.search(combined) for pattern in _NEGATION_PATTERNS)

def compute_confidence(
    trgm_score: float,
    tfidf_score: float,
    importance: float,
    freshness_days: float,
    embedding_score: float | None = None,
    access_count: int = 0,
) -> tuple[float, str]:
    """Compute grounding confidence from multiple signals.

    V4: access_count adaptively boosts importance weight.
    Observations accessed ≥10 times saturate the boost.

    Args:
        trgm_score: Trigram similarity [0, 1].
        tfidf_score: TF-IDF similarity [0, 1].
        importance: Observation importance [0, 1].
        freshness_days: Days since last access.
        embedding_score: Optional embedding cosine similarity.
        access_count: Times this observation was accessed.

    Returns:
        Tuple of (score, level).
    """
    freshness = 1.0 / (1.0 + freshness_days / 30.0)

    # V4: Adaptive importance weight — frequently accessed = more trustworthy
    recall_factor = min(1.0, access_count / 10.0)
    adaptive_importance_w = 0.10 + 0.10 * recall_factor  # 0.10 → 0.20

    if embedding_score is not None:
        score = (
            0.30 * embedding_score
            + 0.25 * trgm_score
            + 0.20 * tfidf_score
            + adaptive_importance_w * importance
            + (0.15 - adaptive_importance_w + 0.10) * freshness
        )
    else:
        score = (
            0.30 * trgm_score
            + 0.25 * tfidf_score
            + 0.20 * (trgm_score * tfidf_score) ** 0.5
            + adaptive_importance_w * importance
            + (0.15 - adaptive_importance_w + 0.10) * freshness
        )

    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        level = "verified"
    elif score >= 0.5:
        level = "partial"
    elif score >= 0.3:
        level = "weak"
    else:
        level = "unknown"

    return round(score, 4), level

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

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
SELECT o.id, o.entity_id, o.content, o.observation_type, o.importance,
    o.source, o.created_at, o.last_accessed, o.access_count,
    e.name AS entity_name, e.entity_type,
    (
        0.35 * ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(o.content, $1) +
        0.25 * o.importance +
        0.10 * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0))
    ) AS score,
    similarity(o.content, $1) AS trgm_similarity,
    EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0 AS days_since_access
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE (o.search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(o.content, $1) > 0.3)
   AND o.observation_type != 'superseded'
   AND (o.valid_until IS NULL OR o.valid_until > NOW())
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

NEIGHBORS_SQL = """
SELECT
    e.name,
    e.entity_type,
    r.relation_type,
    ROUND(r.strength::numeric, 2) AS strength,
    CASE
        WHEN r.from_entity = ANY($1::uuid[]) THEN r.from_entity
        ELSE r.to_entity
    END AS source_entity_id
FROM brain_relations r
JOIN brain_entities e ON (
    CASE
        WHEN r.from_entity = ANY($1::uuid[]) THEN r.to_entity
        ELSE r.from_entity
    END = e.id
)
WHERE r.from_entity = ANY($1::uuid[])
   OR (r.to_entity = ANY($1::uuid[]) AND r.bidirectional = TRUE)
ORDER BY r.strength DESC
LIMIT 30
"""

SEARCH_VECTOR_SQL = """
SELECT o.id, o.content, o.observation_type, o.importance,
    o.source, o.created_at, o.last_accessed, o.access_count,
    e.name AS entity_name, e.entity_type,
    1 - (o.embedding <=> $2::vector) AS vector_score
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE o.embedding IS NOT NULL
    AND o.observation_type != 'superseded'
    AND (o.valid_until IS NULL OR o.valid_until > NOW())
ORDER BY o.embedding <=> $2::vector
LIMIT $1
"""

VERIFY_SQL = """
SELECT
    o.content,
    o.importance,
    o.access_count,
    similarity(o.content, $1) AS trgm_similarity,
    ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1)) AS ts_rank,
    EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0 AS days_since_access,
    e.name AS entity_name
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE (o.search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(o.content, $1) > 0.3)
   AND o.observation_type != 'superseded'
   AND (o.valid_until IS NULL OR o.valid_until > NOW())
ORDER BY similarity(o.content, $1) DESC
LIMIT 5
"""
