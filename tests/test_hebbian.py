"""Tests for hebbian.py — Oja's rule, FSRS decay, and utility functions.

Nemesis Protocol:
  🟢 Normal: standard inputs, expected outputs
  🟡 Pessimistic: boundary values, zeros, negatives
  🔴 Extreme: NaN, Inf, huge values
"""
import math
import pytest

from cuba_memorys.hebbian import (
    ETA,
    MIN_IMPORTANCE,
    MAX_IMPORTANCE,
    oja_positive,
    oja_negative,
    fsrs_retrievability,
    fsrs_update_stability,
    spreading_activation_boost,
    synapse_weight_boost,
    relation_traverse_boost,
    relation_decay,
    transitive_strength,
    information_density,
)


# ── Oja's Rule ──────────────────────────────────────────────────────


class TestOjaPositive:
    """Oja positive: delta = η * (1 - x²), importance + delta."""

    def test_normal_midpoint(self) -> None:
        """🟢 At x=0.5, delta = 0.05 * (1 - 0.25) = 0.0375."""
        result = oja_positive(0.5)
        expected = 0.5 + ETA * (1.0 - 0.5 * 0.5)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_positive_increases(self) -> None:
        """🟢 Positive feedback always increases importance."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert oja_positive(x) > x

    def test_convergence_near_one(self) -> None:
        """🟢 At x→1.0, delta→0 (self-limiting property of Oja's rule)."""
        result = oja_positive(0.99)
        delta = result - 0.99
        assert delta < 0.002  # Very small increment near saturation

    def test_clamped_to_max(self) -> None:
        """🟡 Result never exceeds MAX_IMPORTANCE=1.0."""
        assert oja_positive(1.0) <= MAX_IMPORTANCE

    def test_clamped_to_min(self) -> None:
        """🟡 At MIN_IMPORTANCE, result ≥ MIN_IMPORTANCE."""
        result = oja_positive(MIN_IMPORTANCE)
        assert result >= MIN_IMPORTANCE

    def test_zero_importance(self) -> None:
        """🟡 At x=0: delta = η * 1.0 = 0.05."""
        result = oja_positive(0.0)
        assert result == pytest.approx(ETA, rel=1e-9)


class TestOjaNegative:
    """Oja negative: delta = η * (1 + x²), importance - delta."""

    def test_normal_midpoint(self) -> None:
        """🟢 At x=0.5, delta = 0.05 * (1 + 0.25) = 0.0625."""
        result = oja_negative(0.5)
        expected = 0.5 - ETA * (1.0 + 0.5 * 0.5)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_negative_decreases(self) -> None:
        """🟢 Negative feedback always decreases importance."""
        for x in [0.2, 0.5, 0.8]:
            assert oja_negative(x) < x

    def test_clamped_to_min(self) -> None:
        """🟡 Never goes below MIN_IMPORTANCE."""
        result = oja_negative(MIN_IMPORTANCE)
        assert result >= MIN_IMPORTANCE

    def test_clamped_to_max(self) -> None:
        """🟡 Never exceeds MAX_IMPORTANCE."""
        assert oja_negative(MAX_IMPORTANCE) <= MAX_IMPORTANCE


# ── FSRS — Ye (2023) ────────────────────────────────────────────────


class TestFSRSRetrievability:
    """R(t,S) = (1 + t/(9*S))^(-1). Power-law forgetting curve."""

    def test_zero_elapsed(self) -> None:
        """🟢 At t=0, R=1.0 (perfect recall)."""
        assert fsrs_retrievability(0.0, 10.0) == 1.0

    def test_one_day(self) -> None:
        """🟢 After 1 day with S=9: R = (1+1/81)^(-1) ≈ 0.9878."""
        result = fsrs_retrievability(1.0, 9.0)
        expected = (1.0 + 1.0 / 81.0) ** (-1.0)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_long_elapsed_decays(self) -> None:
        """🟢 After 365 days, recall is significantly reduced."""
        result = fsrs_retrievability(365.0, 10.0)
        assert result < 0.2

    def test_higher_stability_slower_decay(self) -> None:
        """🟢 Higher stability → slower forgetting."""
        r_low = fsrs_retrievability(30.0, 5.0)
        r_high = fsrs_retrievability(30.0, 50.0)
        assert r_high > r_low

    def test_zero_stability(self) -> None:
        """🟡 S=0 → R=1.0 (guard clause)."""
        assert fsrs_retrievability(10.0, 0.0) == 1.0

    def test_negative_stability(self) -> None:
        """🟡 S<0 → R=1.0 (guard clause)."""
        assert fsrs_retrievability(10.0, -5.0) == 1.0

    def test_negative_elapsed(self) -> None:
        """🟡 t<0 → R=1.0 (guard clause)."""
        assert fsrs_retrievability(-1.0, 10.0) == 1.0

    def test_result_bounded(self) -> None:
        """🔴 R always in [0, 1]."""
        for t in [0.0, 1.0, 100.0, 10000.0]:
            for s in [0.1, 1.0, 100.0]:
                r = fsrs_retrievability(t, s)
                assert 0.0 <= r <= 1.0


class TestFSRSUpdateStability:
    """FSRS v4 stability update after successful recall."""

    def test_stability_increases(self) -> None:
        """🟢 After recall, stability increases (memory strengthens)."""
        new_s = fsrs_update_stability(10.0, 5.0, 0.8)
        assert new_s > 10.0

    def test_easy_items_grow_faster(self) -> None:
        """🟢 Low difficulty → larger stability growth."""
        s_easy = fsrs_update_stability(10.0, 2.0, 0.8)
        s_hard = fsrs_update_stability(10.0, 9.0, 0.8)
        assert s_easy > s_hard

    def test_low_retrievability_boosts_more(self) -> None:
        """🟢 Desirable difficulty: low R before recall → more growth."""
        s_low_r = fsrs_update_stability(10.0, 5.0, 0.3)
        s_high_r = fsrs_update_stability(10.0, 5.0, 0.9)
        assert s_low_r > s_high_r


# ── Utility Functions ────────────────────────────────────────────────


class TestSpreadingActivation:
    def test_boosts_importance(self) -> None:
        """🟢 Always increases importance."""
        result = spreading_activation_boost(0.5)
        assert result > 0.5

    def test_clamped_to_max(self) -> None:
        """🟡 Never exceeds MAX_IMPORTANCE."""
        assert spreading_activation_boost(1.0) <= MAX_IMPORTANCE


class TestSynapseWeightBoost:
    def test_increases_weight(self) -> None:
        """🟢 Boost increases weight."""
        result = synapse_weight_boost(1.0)
        assert result > 1.0

    def test_self_limiting_near_max(self) -> None:
        """🟢 Boost diminishes as weight → max_weight (Hebbian saturation)."""
        delta_low = synapse_weight_boost(1.0) - 1.0
        delta_high = synapse_weight_boost(4.5) - 4.5
        assert delta_low > delta_high

    def test_clamped_at_max(self) -> None:
        """🟡 Never exceeds max_weight."""
        assert synapse_weight_boost(5.0, max_weight=5.0) <= 5.0

    def test_custom_max_weight(self) -> None:
        """🟢 Custom max_weight works."""
        result = synapse_weight_boost(9.0, max_weight=10.0)
        assert result > 9.0
        assert result <= 10.0


class TestRelationDecay:
    def test_no_decay_at_zero(self) -> None:
        """🟡 Zero elapsed → no decay."""
        assert relation_decay(0.5, 0.0) == 0.5

    def test_decays_over_time(self) -> None:
        """🟢 Strength decays exponentially."""
        result = relation_decay(1.0, 30.0)
        assert result < 1.0
        assert result > 0.0

    def test_clamped_to_min(self) -> None:
        """🔴 After extreme time, doesn't go below MIN_IMPORTANCE."""
        result = relation_decay(1.0, 100000.0)
        assert result >= MIN_IMPORTANCE


class TestTransitiveStrength:
    def test_product_with_decay(self) -> None:
        """🟢 s_ab * s_bc * 0.9^depth."""
        result = transitive_strength(0.8, 0.6, 1)
        expected = 0.8 * 0.6 * (0.9 ** 1)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_deeper_path_weaker(self) -> None:
        """🟢 Deeper paths have weaker transitive strength."""
        s1 = transitive_strength(0.8, 0.6, 1)
        s3 = transitive_strength(0.8, 0.6, 3)
        assert s1 > s3


class TestInformationDensity:
    def test_empty_text(self) -> None:
        """🟡 Empty → 0."""
        assert information_density("") == 0.0

    def test_single_word(self) -> None:
        """🟡 Single word → 0 (< 2 words guard)."""
        assert information_density("hello") == 0.0

    def test_all_same_words(self) -> None:
        """🟢 Repetitive text → low density."""
        result = information_density("the the the the the")
        assert result == 0.0

    def test_diverse_text(self) -> None:
        """🟢 Diverse text → high density (near 1.0)."""
        result = information_density("python fastapi postgresql docker redis")
        assert result > 0.9

    def test_bounded(self) -> None:
        """🔴 Always in [0, 1]."""
        for text in ["a b c", "x " * 100, "hello world foo bar baz"]:
            r = information_density(text)
            assert 0.0 <= r <= 1.0
