//! Custom FSRS-6 implementation (ADR-001: ~50 LOC, zero deps).
//!
//! Implements retrievability calculation and stability update.
//! Based on Ye 2024 paper, 21 parameters.
//!
//! V4: FSRS-6 upgrade from FSRS-4 (21 params instead of 4).
//! V5: Personalizable decay rate w20 (Expertium 2025) — per-entity.

use crate::constants::{DECAY_THRESHOLD, DEFAULT_DECAY_RATE, FSRS6_DEFAULT_PARAMS};
use anyhow::Result;
use sqlx::PgPool;

/// FSRS-6 default decay constant.
const DECAY: f64 = -0.5;
/// FSRS-6 factor derived from decay constant.
const FACTOR: f64 = 0.9f64; // 19.0 / 81.0 precomputed below

/// Calculate retrievability from stability and elapsed days.
///
/// Formula: R(t, S) = (1 + FACTOR * t / S)^(-decay_rate)
///
/// V5: decay_rate is now personalizable per entity (w20 in FSRS-6).
/// Range: [0.1, 0.8]. Default: 0.5. Lower = slower forgetting.
///
/// Args:
///     stability: Current stability parameter (S).
///     elapsed_days: Days since last review/access.
///
/// Returns:
///     Retrievability in [0.0, 1.0].
pub fn retrievability(stability: f64, elapsed_days: f64) -> f64 {
    retrievability_with_decay(stability, elapsed_days, DEFAULT_DECAY_RATE)
}

/// Retrievability with custom decay rate (FSRS-6 w20).
pub fn retrievability_with_decay(stability: f64, elapsed_days: f64, decay_rate: f64) -> f64 {
    if stability <= 0.0 || elapsed_days < 0.0 {
        return 0.0;
    }
    let factor = 19.0_f64 / 81.0;
    let clamped_decay = decay_rate.clamp(0.1, 0.8);
    (1.0 + factor * elapsed_days / stability).powf(-clamped_decay)
}

/// Compute adaptive decay rate from access_count (FSRS-6 w20).
///
/// High-access entities forget slower; rarely-accessed forget faster.
/// Uses sigmoid mapping: decay = 0.1 + 0.7 / (1 + e^(0.05 * (count - 30)))
///
/// Returns:
///     Decay rate in [0.1, 0.8].
pub fn adaptive_decay_rate(access_count: i32) -> f64 {
    let sigmoid = 1.0 / (1.0 + (0.05 * (access_count as f64 - 30.0)).exp());
    (0.1 + 0.7 * sigmoid).clamp(0.1, 0.8)
}

/// Update stability after a recall event.
///
/// Uses FSRS-6 formula with 21 default parameters.
///
/// Args:
///     current_stability: Current S value.
///     difficulty: Current D value (1-10).
///     retrievability: Current R(t, S).
///     rating: How well recalled (0-3, where 3 = easy).
///
/// Returns:
///     New stability value.
pub fn update_stability(
    current_stability: f64,
    difficulty: f64,
    retrievability: f64,
    rating: u8,
) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;

    // Clamp inputs
    let d = difficulty.clamp(1.0, 10.0);
    let r = retrievability.clamp(0.0, 1.0);
    let s = current_stability.max(0.01);

    match rating {
        // Again (forgot) → new stability from scratch
        0 => {
            w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0)
                * (w[14] * (1.0 - r)).exp()
        }
        // Hard/Good/Easy → multiply current stability
        _ => {
            let rating_bonus = match rating {
                1 => w[15], // Hard
                2 => 1.0,   // Good (baseline)
                3 => w[16], // Easy
                _ => 1.0,
            };
            s * (1.0 + (w[8] * d.powf(-w[9]) * (s.powf(-w[10]) - 1.0) * rating_bonus).exp())
        }
    }
}

/// Update difficulty based on rating.
///
/// Args:
///     current_difficulty: Current D value.
///     rating: How well recalled (0-3).
///
/// Returns:
///     New difficulty value in [1.0, 10.0].
pub fn update_difficulty(current_difficulty: f64, rating: u8) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;
    let d = current_difficulty;

    // Mean reversion toward initial difficulty
    let delta = -(w[6] as f64) * (rating as f64 - 3.0);
    let new_d = d + delta;

    // Apply mean reversion
    let mean_reversion = w[7] * (w[4] - new_d);
    (new_d + mean_reversion).clamp(1.0, 10.0)
}

/// Batch decay — apply FSRS retrievability check on all observations.
///
/// V1: Single batch UPDATE instead of N individual queries.
///
/// Args:
///     pool: Database pool.
///     protected: Entity IDs to skip (active session protection).
///
/// Returns:
///     Number of observations with updated stability.
pub async fn batch_decay(pool: &PgPool, protected: &[uuid::Uuid]) -> Result<usize> {
    // V5: Per-entity adaptive decay rate based on access_count.
    // Uses sigmoid mapping: high-access entities decay slower.
    let result = sqlx::query(
        r#"
        UPDATE brain_observations o SET
            stability = stability * 0.95,
            retrieval_strength = retrieval_strength * 0.95,
            updated_at = NOW()
        FROM brain_entities e
        WHERE o.entity_id = e.id
          AND o.entity_id NOT IN (SELECT UNNEST($1::uuid[]))
          AND o.observation_type != 'superseded'
          AND (
            -- V5: Adaptive decay rate from access_count via sigmoid
            -- decay_rate = clamp(0.1 + 0.7 / (1 + exp(0.05 * (count - 30))), 0.1, 0.8)
            POWER(
                1.0 + (19.0/81.0) * EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0 / GREATEST(o.stability, 0.01),
                -LEAST(0.8, GREATEST(0.1, 0.1 + 0.7 / (1.0 + EXP(0.05 * (e.access_count - 30)))))
            ) < $2
          )
        "#,
    )
    .bind(protected)
    .bind(DECAY_THRESHOLD)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrievability_fresh() {
        let r = retrievability(1.0, 0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_retrievability_decays() {
        let r = retrievability(1.0, 10.0);
        assert!(r < 0.6, "R should decay: got {r}");
        assert!(r > 0.4, "R should not collapse: got {r}");
    }

    #[test]
    fn test_retrievability_high_stability() {
        let r = retrievability(30.0, 10.0);
        assert!(r > 0.8);
    }

    #[test]
    fn test_retrievability_custom_decay_slow() {
        // V5: Low decay (0.1) → forgets much slower
        let r_slow = retrievability_with_decay(1.0, 10.0, 0.1);
        let r_default = retrievability(1.0, 10.0);
        assert!(r_slow > r_default, "slow decay should retain more: {r_slow} vs {r_default}");
    }

    #[test]
    fn test_retrievability_custom_decay_fast() {
        // V5: High decay (0.8) → forgets faster
        let r_fast = retrievability_with_decay(1.0, 10.0, 0.8);
        let r_default = retrievability(1.0, 10.0);
        assert!(r_fast < r_default, "fast decay should retain less: {r_fast} vs {r_default}");
    }

    #[test]
    fn test_adaptive_decay_rate_low_access() {
        // V5: Low access → high decay rate (forgets faster)
        let rate = adaptive_decay_rate(0);
        assert!(rate > 0.5, "low access should have high decay: {rate}");
    }

    #[test]
    fn test_adaptive_decay_rate_high_access() {
        // V5: High access → low decay rate (forgets slower)
        let rate = adaptive_decay_rate(100);
        assert!(rate < 0.2, "high access should have low decay: {rate}");
    }

    #[test]
    fn test_adaptive_decay_rate_bounds() {
        // V5: Always within [0.1, 0.8]
        for count in [0, 1, 5, 10, 30, 50, 100, 500, 1000] {
            let rate = adaptive_decay_rate(count);
            assert!(rate >= 0.1 && rate <= 0.8, "out of bounds at count={count}: {rate}");
        }
    }

    #[test]
    fn test_update_stability_recall() {
        let new_s = update_stability(1.0, 5.0, 0.9, 2);
        assert!(new_s > 1.0, "stability should increase on recall");
    }

    #[test]
    fn test_update_stability_forget() {
        let new_s = update_stability(10.0, 5.0, 0.3, 0);
        assert!(new_s < 10.0, "stability should decrease on forget, got {new_s}");
        assert!(new_s > 0.0, "stability should remain positive");
    }

    #[test]
    fn test_difficulty_bounds() {
        let d = update_difficulty(5.0, 0);
        assert!(d >= 1.0 && d <= 10.0);
    }
}
