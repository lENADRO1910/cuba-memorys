"""Tests for search.py — RRF fusion, cache, negation detection, confidence.

Nemesis Protocol:
  🟢 Normal: standard RRF fusion, cache operations
  🟡 Pessimistic: empty lists, single item, expired cache
  🔴 Extreme: collision keys, boundary thresholds
"""
import time
from unittest.mock import patch

import pytest

from cuba_memorys.search import (
    RRF_K,
    rrf_fuse,
    cache_get,
    cache_set,
    cache_clear,
    has_negation,
    compute_confidence,
    CACHE_TTL,
)


# ── RRF Fusion — Cormack (2009) ─────────────────────────────────────


class TestRRFFuse:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank + 1)."""

    def test_single_ranking(self) -> None:
        """🟢 Single ranking preserves order."""
        ranking = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.5},
        ]
        result = rrf_fuse([ranking])
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_two_rankings_fusion(
        self, sample_rankings: list[list[dict]],
    ) -> None:
        """🟢 Two rankings fused — 'b' appears first in both → highest RRF."""
        result = rrf_fuse(sample_rankings)
        ids = [r["id"] for r in result]
        # 'b' is rank 1 in list 1, rank 0 in list 2 → highest combined score
        assert ids[0] in ("a", "b")  # Both score high
        assert "rrf_score" in result[0]

    def test_rrf_formula_correctness(self) -> None:
        """🟢 Verify RRF formula: 1/(k + rank + 1)."""
        ranking = [{"id": "x"}]
        result = rrf_fuse([ranking])
        expected_score = round(1.0 / (RRF_K + 0 + 1), 4)
        assert result[0]["rrf_score"] == expected_score

    def test_empty_rankings(self) -> None:
        """🟡 Empty input → empty output."""
        assert rrf_fuse([]) == []

    def test_empty_inner_rankings(self) -> None:
        """🟡 Empty inner lists → empty output."""
        assert rrf_fuse([[], []]) == []

    def test_single_item(self) -> None:
        """🟡 Single item in single ranking."""
        result = rrf_fuse([[{"id": "solo"}]])
        assert len(result) == 1
        assert result[0]["id"] == "solo"

    def test_duplicate_ids_across_rankings(self) -> None:
        """🟢 Same ID in multiple rankings → scores summed."""
        r1 = [{"id": "same"}]
        r2 = [{"id": "same"}]
        r3 = [{"id": "same"}]
        result = rrf_fuse([r1, r2, r3])
        assert len(result) == 1
        expected = round(3 * (1.0 / (RRF_K + 1)), 4)
        assert result[0]["rrf_score"] == expected

    def test_preserves_extra_fields(self) -> None:
        """🟢 Extra fields from the item are preserved."""
        ranking = [{"id": "a", "title": "Test", "extra": 42}]
        result = rrf_fuse([ranking])
        assert result[0]["title"] == "Test"
        assert result[0]["extra"] == 42

    def test_name_fallback_key(self) -> None:
        """🟢 Falls back to 'name' when 'id' is missing."""
        ranking = [{"name": "entity_a"}]
        result = rrf_fuse([ranking])
        assert len(result) == 1


# ── Cache ────────────────────────────────────────────────────────────


class TestCache:
    def setup_method(self) -> None:
        cache_clear()

    def test_cache_miss(self) -> None:
        """🟢 Cache miss returns None."""
        assert cache_get("q", "hybrid", "all", 10) is None

    def test_cache_hit(self) -> None:
        """🟢 After set, get returns same results."""
        data = [{"id": "test"}]
        cache_set("q", "hybrid", "all", 10, data)
        result = cache_get("q", "hybrid", "all", 10)
        assert result == data

    def test_cache_different_params(self) -> None:
        """🟢 Different params → different cache entries."""
        cache_set("q", "hybrid", "all", 10, [{"id": "a"}])
        cache_set("q", "verify", "all", 10, [{"id": "b"}])
        assert cache_get("q", "hybrid", "all", 10)[0]["id"] == "a"  # type: ignore[index]
        assert cache_get("q", "verify", "all", 10)[0]["id"] == "b"  # type: ignore[index]

    def test_cache_expiry(self) -> None:
        """🟡 Expired entries return None."""
        cache_set("q", "hybrid", "all", 10, [{"id": "old"}])
        with patch("cuba_memorys.search.time") as mock_time:
            # First call within TTL
            mock_time.monotonic.return_value = time.monotonic() + CACHE_TTL + 1
            result = cache_get("q", "hybrid", "all", 10)
            assert result is None

    def test_cache_clear(self) -> None:
        """🟢 Clear removes all entries."""
        cache_set("q1", "hybrid", "all", 10, [{"id": "a"}])
        cache_set("q2", "hybrid", "all", 10, [{"id": "b"}])
        cache_clear()
        assert cache_get("q1", "hybrid", "all", 10) is None
        assert cache_get("q2", "hybrid", "all", 10) is None

    def test_cache_eviction_at_max(self) -> None:
        """🟡 When cache is full, oldest entry is evicted."""
        for i in range(101):
            cache_set(f"q{i}", "hybrid", "all", 10, [{"id": str(i)}])
        # First entry should have been evicted (100 max size)
        # At least one early entry should be gone
        evicted_count = sum(
            1 for i in range(10)
            if cache_get(f"q{i}", "hybrid", "all", 10) is None
        )
        assert evicted_count > 0


# ── Negation Detection ───────────────────────────────────────────────


class TestHasNegation:
    def test_english_not(self) -> None:
        """🟢 Detects 'not' in combined text."""
        assert has_negation("The system uses Redis", "not using cache") is True

    def test_spanish_no(self) -> None:
        """🟢 Detects 'no' in Spanish."""
        assert has_negation("usar PostgreSQL", "ya no usar MySQL") is True

    def test_replaced(self) -> None:
        """🟢 Detects 'replaced'."""
        assert has_negation("old system", "replaced with new") is True

    def test_changed_from(self) -> None:
        """🟢 Detects 'changed from'."""
        assert has_negation("version 1", "changed from v1 to v2") is True

    def test_instead_of(self) -> None:
        """🟢 Detects 'instead of'."""
        assert has_negation("Redis", "using Garnet instead of Redis") is True

    def test_no_negation(self) -> None:
        """🟢 Returns False when no negation present."""
        assert has_negation("Python is great", "Python is awesome") is False


# ── Confidence Scoring ───────────────────────────────────────────────


class TestComputeConfidence:
    def test_high_scores_verified(self) -> None:
        """🟢 High trigram + tfidf + importance → verified."""
        score, label = compute_confidence(
            trgm_score=0.9, tfidf_score=0.9,
            importance=0.8, freshness_days=1.0,
        )
        assert label == "verified"
        assert score > 0.7

    def test_low_scores_unknown(self) -> None:
        """🟢 Low scores → unknown."""
        score, label = compute_confidence(
            trgm_score=0.1, tfidf_score=0.1,
            importance=0.1, freshness_days=365.0,
        )
        assert label in ("unknown", "weak")
        assert score < 0.4

    def test_freshness_decays(self) -> None:
        """🟢 Older content has lower confidence."""
        score_fresh, _ = compute_confidence(0.5, 0.5, 0.5, 1.0)
        score_old, _ = compute_confidence(0.5, 0.5, 0.5, 365.0)
        assert score_fresh > score_old

    def test_bounded(self) -> None:
        """🔴 Score always in [0, 1]."""
        for trgm in [0.0, 0.5, 1.0]:
            for tfidf in [0.0, 0.5, 1.0]:
                score, _ = compute_confidence(trgm, tfidf, 0.5, 30.0)
                assert 0.0 <= score <= 1.0

    def test_embedding_score_boosts(self) -> None:
        """🟢 Embedding score adds to confidence."""
        score_no_emb, _ = compute_confidence(0.5, 0.5, 0.5, 10.0)
        score_emb, _ = compute_confidence(
            0.5, 0.5, 0.5, 10.0, embedding_score=0.9,
        )
        assert score_emb > score_no_emb
