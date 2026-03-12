//! Handler: cuba_faro — Hybrid search with RRF fusion.
//!
//! FIX B2: embed() runs in spawn_blocking (not blocking event loop).
//! §A: Weighted RRF Entropy Routing — dynamic weight based on query entropy.
//! V8: Graceful degradation — if vector search fails, fallback to text.
//! VF2: Testing Effect — search matches boost retrieval_strength.

use crate::constants::{RRF_K_MIN, RRF_K_MAX};
use crate::cognitive::dual_strength;
use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};

const DEFAULT_LIMIT: i64 = 10;
const MAX_LIMIT: i64 = 50;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("hybrid");
    let scope = args.get("scope").and_then(|v| v.as_str()).unwrap_or("all");
    let limit = args.get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_LIMIT)
        .min(MAX_LIMIT);

    match mode {
        "hybrid" => hybrid_search(pool, query, scope, limit).await,
        "verify" => verify_claim(pool, query).await,
        _ => anyhow::bail!("Invalid mode: {mode}. Use hybrid/verify"),
    }
}

/// §A V2: Weighted RRF Entropy Routing — 3 ranges (Elastic 2025).
async fn hybrid_search(pool: &PgPool, query: &str, scope: &str, limit: i64) -> Result<Value> {
    // §A V2: 3-range entropy routing (keyword / mixed / semantic)
    let query_entropy = compute_query_entropy(query);
    let (text_weight, vector_weight) = entropy_weights(query_entropy);

    // Run text search (always available — V8 graceful degradation base)
    let text_results = text_search(pool, query, scope, limit * 2).await?;

    // Run vector search (may fail gracefully — V8)
    let vector_results = vector_search(pool, query, scope, limit * 2).await;

    // P3: Adaptive RRF k — scales with result count (Azure AI Search 2025)
    let total_results = text_results.len()
        + vector_results.as_ref().map(|v| v.len()).unwrap_or(0);
    let rrf_k = adaptive_rrf_k(total_results);

    // RRF Fusion with entropy-weighted scores and adaptive k
    let mut fused_scores: HashMap<String, (f64, Value)> = HashMap::new();

    // Add text results with RRF rank score
    for (rank, result) in text_results.iter().enumerate() {
        let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let rrf_score = text_weight / (rrf_k + rank as f64 + 1.0);
        fused_scores.insert(id, (rrf_score, result.clone()));
    }

    // Add vector results (V8: only if available)
    if let Ok(vec_results) = vector_results {
        for (rank, result) in vec_results.iter().enumerate() {
            let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let rrf_score = vector_weight / (rrf_k + rank as f64 + 1.0);
            fused_scores
                .entry(id.clone())
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert((rrf_score, result.clone()));
        }
    }

    // Sort by fused score
    let mut results: Vec<(String, f64, Value)> = fused_scores
        .into_iter()
        .map(|(id, (score, result))| (id, score, result))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit as usize);

    // VF2: Testing Effect — boost retrieval_strength on matched observations
    let matched_obs_ids: Vec<uuid::Uuid> = results.iter()
        .filter_map(|(_, _, r)| {
            r.get("id").and_then(|v| v.as_str())
                .and_then(|s| s.parse::<uuid::Uuid>().ok())
        })
        .collect();

    if !matched_obs_ids.is_empty() {
        if let Err(e) = dual_strength::on_search_match(pool, &matched_obs_ids).await {
            tracing::warn!(error = %e, "failed to apply Testing Effect boost");
        }
    }

    // Session awareness: check active session and boost matching results
    let session_boost = get_session_goals(pool).await.unwrap_or_default();
    if !session_boost.is_empty() {
        for (_, score, result) in &mut results {
            if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
                let content_lower = content.to_lowercase();
                for goal in &session_boost {
                    if content_lower.contains(&goal.to_lowercase()) {
                        *score *= 1.3; // 30% boost for session-relevant results
                        break;
                    }
                }
            }
        }
        // Re-sort after session boost
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    let results_json: Vec<Value> = results
        .iter()
        .map(|(_, score, result)| {
            let mut r = result.clone();
            if let Some(obj) = r.as_object_mut() {
                obj.insert("fused_score".to_string(), serde_json::json!(score));
            }
            r
        })
        .collect();

    Ok(serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": results_json,
        "count": results_json.len(),
        "search_config": {
            "query_entropy": query_entropy,
            "text_weight": text_weight,
            "vector_weight": vector_weight,
            "rrf_k": rrf_k,
            "session_aware": !session_boost.is_empty()
        }
    }))
}

/// Verify a claim against stored knowledge.
async fn verify_claim(pool: &PgPool, claim: &str) -> Result<Value> {
    let evidence: Vec<(String, f64, String)> = sqlx::query_as(
        "SELECT content, similarity(content, $1)::float8 AS sim, observation_type
         FROM brain_observations
         WHERE similarity(content, $1) > 0.3
           AND observation_type != 'superseded'
         ORDER BY sim DESC
         LIMIT 10"
    )
    .bind(claim)
    .fetch_all(pool)
    .await?;

    let (confidence, level) = if evidence.is_empty() {
        (0.0, "unknown")
    } else {
        let max_sim = evidence.first().map(|(_, s, _)| *s).unwrap_or(0.0);
        let avg_sim = evidence.iter().map(|(_, s, _)| s).sum::<f64>() / evidence.len() as f64;
        let count = evidence.len();

        let confidence = max_sim * 0.5 + avg_sim * 0.3 + (count as f64 / 10.0).min(1.0) * 0.2;
        let level = if confidence > 0.7 {
            "verified"
        } else if confidence > 0.5 {
            "partial"
        } else if confidence > 0.3 {
            "weak"
        } else {
            "unknown"
        };
        (confidence, level)
    };

    let evidence_json: Vec<Value> = evidence
        .iter()
        .map(|(content, sim, obs_type)| {
            serde_json::json!({
                "content": content,
                "similarity": sim,
                "type": obs_type
            })
        })
        .collect();

    Ok(serde_json::json!({
        "mode": "verify",
        "claim": claim,
        "confidence": confidence,
        "grounding": level,
        "evidence": evidence_json,
        "evidence_count": evidence_json.len()
    }))
}

/// Execute a parameterized text search query and map results to JSON via closure.
async fn search_table<T, F>(
    pool: &PgPool,
    sql: &str,
    query: &str,
    limit: i64,
    mapper: F,
) -> Result<Vec<Value>>
where
    T: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    F: Fn(T) -> Value,
{
    let rows: Vec<T> = sqlx::query_as(sql)
        .bind(query)
        .bind(limit)
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(mapper).collect())
}

/// Text search using PostgreSQL full-text + trigram similarity.
async fn text_search(pool: &PgPool, query: &str, scope: &str, limit: i64) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    if scope == "all" || scope == "entities" {
        let entities = search_table::<(uuid::Uuid, String, String, f64, f64), _>(
            pool,
            "SELECT id, name, entity_type, importance::float8,
                    (ts_rank(search_vector, plainto_tsquery('simple', $1)) +
                    similarity(name, $1))::float8 AS score
             FROM brain_entities
             WHERE search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(name, $1) > 0.3
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, name, entity_type, importance, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "entity",
                    "name": name,
                    "entity_type": entity_type,
                    "importance": importance,
                    "score": score
                })
            },
        ).await?;
        results.extend(entities);
    }

    if scope == "all" || scope == "observations" {
        let observations = search_table::<(uuid::Uuid, String, String, String, f64, f64), _>(
            pool,
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8,
                    (ts_rank(o.search_vector, plainto_tsquery('simple', $1)) +
                    similarity(o.content, $1))::float8 AS score
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE (o.search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(o.content, $1) > 0.3)
               AND o.observation_type != 'superseded'
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, entity_name, content, obs_type, importance, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "observation",
                    "entity_name": entity_name,
                    "content": content,
                    "observation_type": obs_type,
                    "importance": importance,
                    "score": score
                })
            },
        ).await?;
        results.extend(observations);
    }

    if scope == "all" || scope == "errors" {
        let errors = search_table::<(uuid::Uuid, String, String, bool, f64), _>(
            pool,
            "SELECT id, error_type, error_message, resolved,
                    (ts_rank(search_vector, plainto_tsquery('simple', $1)) +
                    similarity(error_message, $1))::float8 AS score
             FROM brain_errors
             WHERE search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(error_message, $1) > 0.3
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, error_type, error_message, resolved, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "error",
                    "error_type": error_type,
                    "error_message": error_message,
                    "resolved": resolved,
                    "score": score
                })
            },
        ).await?;
        results.extend(errors);
    }

    Ok(results)
}

/// Vector search via pgvector cosine similarity.
/// V8: Returns Result — callers can gracefully degrade on failure.
async fn vector_search(pool: &PgPool, query: &str, _scope: &str, limit: i64) -> Result<Vec<Value>> {
    // Compute embedding via ONNX (or hash fallback).
    // embed() already runs in spawn_blocking internally (FIX B2).
    let embedding = match crate::embeddings::onnx::embed(query).await {
        Ok(emb) => emb,
        Err(_) => return Ok(vec![]), // V8: graceful degradation
    };

    // Skip if embedding is all zeros (no model loaded)
    if embedding.iter().all(|&v| v == 0.0) {
        return Ok(vec![]); // V8: graceful degradation
    }

    // If we had real embeddings:
    let observations: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
        "SELECT o.id, e.name, o.content,
                1.0 - (o.embedding <=> $1::vector) AS cosine_sim
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.embedding IS NOT NULL
           AND o.observation_type != 'superseded'
         ORDER BY o.embedding <=> $1::vector
         LIMIT $2"
    )
    .bind(pgvector::Vector::from(embedding))
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let results: Vec<Value> = observations
        .iter()
        .map(|(id, entity_name, content, sim)| {
            serde_json::json!({
                "id": id.to_string(),
                "type": "observation",
                "entity_name": entity_name,
                "content": content,
                "cosine_similarity": sim
            })
        })
        .collect();

    Ok(results)
}

// ── Utility Functions ───────────────────────────────────────────

/// §A V2: 3-range entropy routing (Elastic Search Labs, 2025).
///
/// | Entropy | Query Type | text | vector |
/// |---------|------------|------|--------|
/// | < 2.0   | Keyword    | 0.7  | 0.3    |
/// | 2.0-3.5 | Mixed      | 0.5  | 0.5    |
/// | > 3.5   | Semantic   | 0.3  | 0.7    |
fn entropy_weights(entropy: f64) -> (f64, f64) {
    if entropy < 2.0 {
        (0.7, 0.3) // Keyword-heavy: prefer text match
    } else if entropy <= 3.5 {
        (0.5, 0.5) // Balanced: equal weight
    } else {
        (0.3, 0.7) // Semantic-heavy: prefer vector search
    }
}

/// P3: Adaptive RRF k — scales with result count (Azure AI Search 2025).
///
/// With few results (< 20), a lower k prioritizes top-ranked items.
/// With many results (> 120), k = 60 maintains stability.
fn adaptive_rrf_k(result_count: usize) -> f64 {
    (result_count as f64 * 0.5).clamp(RRF_K_MIN, RRF_K_MAX)
}

/// §A: Compute query entropy for RRF weighting.
fn compute_query_entropy(query: &str) -> f64 {
    let words: Vec<&str> = query.split_whitespace().collect();
    let total = words.len();
    if total == 0 {
        return 0.0;
    }

    let unique: HashSet<&str> = words.iter().copied().collect();
    let mut entropy = 0.0;
    for word in &unique {
        let freq = words.iter().filter(|w| *w == word).count() as f64 / total as f64;
        if freq > 0.0 {
            entropy -= freq * freq.log2();
        }
    }
    entropy
}

/// Get active session goals for session-aware boosting.
async fn get_session_goals(pool: &PgPool) -> Result<Vec<String>> {
    let row: Option<(serde_json::Value,)> = sqlx::query_as(
        "SELECT goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
    )
    .fetch_optional(pool)
    .await?;

    if let Some((goals,)) = row {
        let goals: Vec<String> = serde_json::from_value(goals).unwrap_or_default();
        Ok(goals)
    } else {
        Ok(vec![])
    }
}
