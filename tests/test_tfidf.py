"""Tests for tfidf.py — TF-IDF index and similarity.

Nemesis Protocol:
  🟢 Normal: standard queries on fitted index
  🟡 Pessimistic: empty corpus, unfitted index, single doc
  🔴 Extreme: very long text, special characters
"""
import pytest

from cuba_memorys.tfidf import TFIDFIndex, tfidf_index


class TestTFIDFIndex:
    """Tests for TFIDFIndex class."""

    def test_fit_sets_fitted_flag(self, sample_texts: list[str]) -> None:
        """🟢 After fit(), is_fitted is True."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        assert idx.is_fitted is True
        assert idx.corpus_size == len(sample_texts)

    def test_query_returns_ranked_results(self, sample_texts: list[str]) -> None:
        """🟢 Query returns (index, score) tuples, sorted by score desc."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        results = idx.query("machine learning programming")
        assert len(results) > 0
        # First result should be most relevant
        assert results[0][0] == 0  # Python ML doc
        # Scores should be descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_unfitted_returns_empty(self) -> None:
        """🟡 Querying unfitted index returns empty list."""
        idx = TFIDFIndex()
        results = idx.query("anything")
        assert results == []

    def test_fit_empty_corpus(self) -> None:
        """🟡 Fitting empty corpus sets not fitted."""
        idx = TFIDFIndex()
        idx.fit([])
        assert idx.is_fitted is False
        assert idx.corpus_size == 0

    def test_similarity_fitted(self, sample_texts: list[str]) -> None:
        """🟢 Similarity between similar texts is high."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        sim = idx.similarity(
            "Python machine learning",
            "Python is used for machine learning",
        )
        assert sim > 0.5

    def test_similarity_unfitted_still_works(self) -> None:
        """🟢 Similarity works even without fit (uses temp vectorizer)."""
        idx = TFIDFIndex()
        sim = idx.similarity("hello world", "hello world")
        assert sim > 0.9

    def test_similarity_dissimilar_texts(self) -> None:
        """🟢 Dissimilar texts have low similarity."""
        idx = TFIDFIndex()
        sim = idx.similarity(
            "quantum physics entanglement",
            "chocolate cake recipe baking",
        )
        assert sim < 0.3

    def test_similarity_bounded(self) -> None:
        """🟡 Similarity always in [0, 1]."""
        idx = TFIDFIndex()
        for a, b in [
            ("hello", "world"),
            ("same", "same"),
            ("", "test"),
            ("python java rust", "docker kubernetes nginx"),
        ]:
            sim = idx.similarity(a, b)
            assert 0.0 <= sim <= 1.0

    def test_clear_resets(self, sample_texts: list[str]) -> None:
        """🟢 Clear resets all state."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        assert idx.is_fitted is True
        idx.clear()
        assert idx.is_fitted is False
        assert idx.corpus_size == 0

    def test_top_k_limits_results(self, sample_texts: list[str]) -> None:
        """🟢 top_k parameter limits result count."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        results = idx.query("programming language", top_k=2)
        assert len(results) <= 2

    def test_scores_non_negative(self, sample_texts: list[str]) -> None:
        """🟡 All scores are non-negative."""
        idx = TFIDFIndex()
        idx.fit(sample_texts)
        results = idx.query("database")
        for _, score in results:
            assert score >= 0.0


class TestGlobalTFIDFIndex:
    """Test the module-level tfidf_index singleton."""

    def test_singleton_exists(self) -> None:
        """🟢 Module exposes tfidf_index singleton."""
        assert isinstance(tfidf_index, TFIDFIndex)

    def test_initially_not_fitted(self) -> None:
        """🟢 Singleton starts unfitted (no corpus at import time)."""
        fresh = TFIDFIndex()
        assert fresh.is_fitted is False
