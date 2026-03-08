"""Hebbian learning module: Oja's rule + Ebbinghaus decay + Spreading Activation.

Mathematical foundations:
- Oja's Rule (1982): Δw = η·(x·y - y²·w) — prevents weight explosion
- Ebbinghaus (1885): R(t) = e^(-λt), λ = ln(2)/30 ≈ 0.0231 — 30-day half-life
- Collins & Loftus (1975): Spreading activation with 30% transmission per hop
"""

import math

# Learning rate for Oja's rule
ETA: float = 0.05

# Decay constant: half-life = 30 days → λ = ln(2) / 30
DECAY_LAMBDA: float = math.log(2) / 30.0  # ≈ 0.0231

# Spreading activation decay factor per hop
SPREAD_DECAY: float = 0.3

# Minimum importance (never reaches zero)
MIN_IMPORTANCE: float = 0.01

# Maximum importance
MAX_IMPORTANCE: float = 1.0


def oja_positive(importance: float) -> float:
    """Apply positive Hebbian reinforcement using Oja's rule.

    Formula: new_w = min(1.0, w + η·(1 - w²))
    The (1 - w²) term prevents explosion: as w→1, Δw→0.

    Args:
        importance: Current importance weight in [0, 1].

    Returns:
        New importance, guaranteed in [MIN_IMPORTANCE, MAX_IMPORTANCE].
    """
    delta = ETA * (1.0 - importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance + delta))


def oja_negative(importance: float) -> float:
    """Apply negative Hebbian reinforcement.

    Formula: new_w = max(0.01, w - η·(1 + w²))
    Always decreases, faster for higher weights.

    Args:
        importance: Current importance weight in [0, 1].

    Returns:
        New importance, guaranteed in [MIN_IMPORTANCE, MAX_IMPORTANCE].
    """
    delta = ETA * (1.0 + importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance - delta))


def ebbinghaus_decay(importance: float, days_elapsed: float) -> float:
    """Apply Ebbinghaus forgetting curve decay.

    Formula: R(t) = R₀ · e^(-λt)
    With 30-day half-life: after 30 days, importance ≈ 50%.

    Args:
        importance: Current importance.
        days_elapsed: Days since last access.

    Returns:
        Decayed importance, minimum MIN_IMPORTANCE.
    """
    if days_elapsed <= 0:
        return importance
    decayed = importance * math.exp(-DECAY_LAMBDA * days_elapsed)
    return max(MIN_IMPORTANCE, decayed)


def spreading_activation_boost(current_importance: float) -> float:
    """Calculate boost for a neighbor entity via spreading activation.

    When entity X is accessed, its neighbors get boosted by:
    boost = access_boost × SPREAD_DECAY
    Where access_boost = 0.02 (the boost X itself gets).

    Args:
        current_importance: Neighbor's current importance.

    Returns:
        New importance after boost, capped at MAX_IMPORTANCE.
    """
    boost = 0.02 * SPREAD_DECAY  # 0.006
    return min(MAX_IMPORTANCE, current_importance + boost)


def synapse_weight_boost(current_weight: float, max_weight: float = 5.0) -> float:
    """Boost error synapse weight with saturation.

    Formula: Δw = 0.1 · (1 - w/max_w) — slows as approaching max.

    Args:
        current_weight: Current synapse weight.
        max_weight: Maximum allowed weight.

    Returns:
        New synapse weight.
    """
    delta = 0.1 * (1.0 - current_weight / max_weight)
    return min(max_weight, max(0.0, current_weight + delta))


def transitive_strength(
    strength_ab: float, strength_bc: float, depth: int
) -> float:
    """Calculate inferred strength for transitive relation A→B→C.

    Formula: s(A→C) = s(A→B) × s(B→C) × 0.9^depth
    At depth 5: max possible = 0.9⁵ = 0.59.

    Args:
        strength_ab: Strength of first edge.
        strength_bc: Strength of second edge.
        depth: Current traversal depth.

    Returns:
        Inferred strength, always < 1.0.
    """
    return strength_ab * strength_bc * (0.9 ** depth)
