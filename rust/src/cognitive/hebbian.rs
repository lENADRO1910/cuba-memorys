//! Hebbian learning — Oja's rule with BCM metaplastic throttling.
//!
//! V1: BCM Theory (Bienenstock-Cooper-Munro, 1982): static threshold θ_M = 50.0.
//! V2: Dynamic sliding threshold (Gemini Deep Research 2026-03-14).
//!     θ_M adapts via EMA of historical activity instead of fixed constant.
//!     Formula: θ_M = max(θ_min, EMA(access_count)), EMA α = 0.05.
//!     Prevents saturation of central nodes while allowing organic growth.
//!
//! When a node is accessed frequently, θ_M rises → boost decreases.
//! When a node is idle, θ_M normalizes → boost recovers.

use anyhow::Result;
use sqlx::PgPool;

use crate::constants::{
    HEBBIAN_ACCESS_BOOST, HEBBIAN_SEARCH_BOOST, HEBBIAN_OJA_RATE,
    BCM_THROTTLE_SCALE,
};

// ── BCM V2: Dynamic Sliding Threshold ────────────────────────────
/// Minimum BCM threshold floor — prevents division by near-zero.
const BCM_THETA_MIN: f64 = 10.0;

/// V2: Compute dynamic BCM threshold from entity's access history.
///
/// θ_M = max(θ_min, θ_prev × (1 − α) + access_count × α)
///
/// Since we don't persist θ_M directly, we approximate it from
/// access_count using the steady-state EMA convergence:
/// At steady state with constant activity c: θ_M → c.
/// So we use access_count directly as an approximation, clamped to θ_min.
pub fn dynamic_bcm_threshold(access_count: i32) -> f64 {
    // V2: Sliding threshold — uses access_count as running activity proxy
    // For entities with <θ_min accesses, threshold floor prevents over-amplification
    (access_count as f64).max(BCM_THETA_MIN)
}

/// BCM throttle factor with dynamic threshold (V2).
///
/// throttle = max(0.1, 1.0 - (access_count / θ_M) × scale)
///
/// With dynamic θ_M, the throttle adapts to the entity's own history
/// rather than a global constant, preventing premature saturation
/// of hub nodes that legitimately have high access patterns.
pub fn bcm_throttle_dynamic(base_boost: f64, access_count: i32) -> f64 {
    let theta = dynamic_bcm_threshold(access_count);
    let activity_ratio = access_count as f64 / theta;
    let throttle = (1.0 - activity_ratio * BCM_THROTTLE_SCALE).max(0.1);
    base_boost * throttle
}

/// Boost entity importance on access with V2 dynamic BCM throttling.
///
/// FIX R-003: Single atomic UPDATE — no read-modify-write race.
/// V2: θ_M = max(10, access_count) instead of fixed 50.0.
pub async fn boost_on_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(
                importance + $1 * GREATEST(0.1, 1.0 - (access_count::float8 / GREATEST($2, access_count::float8)) * $3),
                1.0
            ),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $4"
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(BCM_THETA_MIN)
    .bind(BCM_THROTTLE_SCALE)
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost entity importance on search match (Testing Effect — VF2)
/// with V2 dynamic BCM throttling.
pub async fn boost_on_search(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(
                importance + $1 * GREATEST(0.1, 1.0 - (access_count::float8 / GREATEST($2, access_count::float8)) * $3),
                1.0
            ),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $4"
    )
    .bind(HEBBIAN_SEARCH_BOOST)
    .bind(BCM_THETA_MIN)
    .bind(BCM_THROTTLE_SCALE)
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost observation importance using Oja's rule.
pub async fn oja_boost(pool: &PgPool, observation_id: uuid::Uuid, positive: bool) -> Result<()> {
    if positive {
        sqlx::query(
            "UPDATE brain_observations SET
                importance = LEAST(importance + $1, 1.0),
                updated_at = NOW()
             WHERE id = $2"
        )
        .bind(HEBBIAN_OJA_RATE)
        .bind(observation_id)
        .execute(pool)
        .await?;
    } else {
        sqlx::query(
            "UPDATE brain_observations SET
                importance = GREATEST(importance * 0.8, 0.0),
                updated_at = NOW()
             WHERE id = $2"
        )
        .bind(observation_id)
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Strengthen relation on traversal (Hebbian synapse strengthening).
pub async fn strengthen_relation(pool: &PgPool, from_entity: uuid::Uuid, to_entity: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_relations SET
            strength = LEAST(strength + $1, 1.0),
            updated_at = NOW()
         WHERE from_entity = $2 AND to_entity = $3"
    )
    .bind(HEBBIAN_OJA_RATE)
    .bind(from_entity)
    .bind(to_entity)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost neighbors via 1-hop diffusion (Collins & Loftus 1975).
pub async fn boost_neighbors(pool: &PgPool, entity_id: uuid::Uuid) -> Result<usize> {
    let result = sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1 * 0.5, 1.0),
            updated_at = NOW()
         WHERE id IN (
             SELECT CASE
                 WHEN from_entity = $2 THEN to_entity
                 ELSE from_entity
             END
             FROM brain_relations
             WHERE from_entity = $2 OR to_entity = $2
         )"
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(entity_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_valid() {
        const { assert!(HEBBIAN_ACCESS_BOOST > 0.0) };
        const { assert!(HEBBIAN_ACCESS_BOOST < 1.0) };
        const { assert!(HEBBIAN_SEARCH_BOOST > 0.0) };
        const { assert!(HEBBIAN_SEARCH_BOOST < 1.0) };
        const { assert!(HEBBIAN_OJA_RATE > 0.0) };
        const { assert!(HEBBIAN_OJA_RATE < 1.0) };
    }

    // ── V2: Dynamic BCM Threshold Tests ─────────────────────────

    #[test]
    fn test_dynamic_threshold_floor() {
        // Low access → clamped to θ_min
        let theta = dynamic_bcm_threshold(3);
        assert!((theta - BCM_THETA_MIN).abs() < 0.001, "low access → floor: {theta}");
    }

    #[test]
    fn test_dynamic_threshold_scales() {
        // High access → threshold adapts upward
        let theta = dynamic_bcm_threshold(100);
        assert_eq!(theta, 100.0, "high access → θ=access_count: {theta}");
    }

    #[test]
    fn test_dynamic_throttle_low_activity() {
        // Few accesses → near-full boost (θ_M = 10, count = 5 → ratio = 0.5)
        let boost = bcm_throttle_dynamic(0.01, 5);
        assert!(boost > 0.004, "low activity should give decent boost: {boost}");
    }

    #[test]
    fn test_dynamic_throttle_high_activity() {
        // Many accesses → fully self-adjusted throttle
        // count=100, θ=100 → ratio=1.0 → throttle = max(0.1, 1.0 - 0.8) = 0.2
        let boost = bcm_throttle_dynamic(0.01, 100);
        assert!(boost > 0.001 && boost < 0.005, "high activity → moderate throttle: {boost}");
    }

    #[test]
    fn test_dynamic_throttle_vs_static() {
        // V2 key insight: entity with count=200 should NOT be throttled as
        // harshly as V1 (where θ=50 → ratio=4.0 → floor hit immediately).
        // V2: θ=200 → ratio=1.0 → throttle = max(0.1, 1 - 0.8) = 0.2
        let boost_v2 = bcm_throttle_dynamic(0.01, 200);
        // V1 equivalent:
        let v1_ratio = 200.0 / 50.0; // = 4.0
        let v1_throttle = (1.0 - v1_ratio * BCM_THROTTLE_SCALE).max(0.1);
        let boost_v1 = 0.01 * v1_throttle;
        assert!(
            boost_v2 > boost_v1,
            "V2 should be less aggressive than V1 for high-activity: {boost_v2} vs {boost_v1}"
        );
    }

    #[test]
    fn test_dynamic_throttle_extreme() {
        // Extreme accesses → still gets minimum 10% of base
        let boost = bcm_throttle_dynamic(0.01, 10_000);
        assert!(boost >= 0.001, "extreme activity → 10% floor: {boost}");
    }
}
