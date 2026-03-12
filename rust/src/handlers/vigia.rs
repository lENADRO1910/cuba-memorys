//! Handler: cuba_vigia — Knowledge graph analytics.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let metric = args.get("metric").and_then(|v| v.as_str()).unwrap_or("summary");

    match metric {
        "summary" => {
            let entities: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_entities").fetch_one(pool).await?;
            let observations: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_observations WHERE observation_type != 'superseded'").fetch_one(pool).await?;
            let relations: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_relations").fetch_one(pool).await?;
            let errors: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_errors").fetch_one(pool).await?;
            let sessions: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_sessions").fetch_one(pool).await?;
            let token_estimate = observations.0 * 50; // ~50 tokens per observation
            Ok(serde_json::json!({"metric": "summary", "entities": entities.0, "observations": observations.0, "relations": relations.0, "errors": errors.0, "sessions": sessions.0, "estimated_tokens": token_estimate}))
        }
        "health" => {
            let avg_importance: (Option<f64>,) = sqlx::query_as("SELECT AVG(importance) FROM brain_entities").fetch_one(pool).await?;
            let stale_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_observations WHERE last_accessed < NOW() - INTERVAL '30 days'").fetch_one(pool).await?;
            let db_size: (String,) = sqlx::query_as("SELECT pg_size_pretty(pg_database_size(current_database()))").fetch_one(pool).await?;
            Ok(serde_json::json!({"metric": "health", "avg_importance": avg_importance.0, "stale_observations": stale_count.0, "database_size": db_size.0}))
        }
        "drift" => {
            let recent_errors: Vec<(String, i64)> = sqlx::query_as(
                "SELECT error_type, COUNT(*) FROM brain_errors WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY error_type ORDER BY COUNT(*) DESC LIMIT 10"
            ).fetch_all(pool).await?;
            let drift: Vec<Value> = recent_errors.iter().map(|(t, c)| serde_json::json!({"error_type": t, "count": c})).collect();
            Ok(serde_json::json!({"metric": "drift", "error_distribution": drift}))
        }
        "communities" => {
            // Leiden community detection (Traag et al., Nature 2019)
            match crate::graph::community::detect(pool).await {
                Ok(communities) => {
                    let community_json: Vec<Value> = communities.iter().map(|(id, members)| {
                        serde_json::json!({
                            "community_id": id,
                            "size": members.len(),
                            "members": members
                        })
                    }).collect();
                    Ok(serde_json::json!({"metric": "communities", "algorithm": "leiden", "communities": community_json, "count": community_json.len()}))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Leiden community detection failed, using fallback");
                    let components: Vec<(String, i64)> = sqlx::query_as(
                        "SELECT e.entity_type, COUNT(*) FROM brain_entities e GROUP BY e.entity_type ORDER BY COUNT(*) DESC"
                    ).fetch_all(pool).await?;
                    let communities: Vec<Value> = components.iter().map(|(t, c)| serde_json::json!({"type": t, "size": c})).collect();
                    Ok(serde_json::json!({"metric": "communities", "algorithm": "fallback_groupby", "communities": communities}))
                }
            }
        }
        "bridges" => {
            // Entities with highest betweenness (most connections across communities)
            let bridges: Vec<(String, i64)> = sqlx::query_as(
                "SELECT e.name, COUNT(r.id) as connection_count FROM brain_entities e
                 LEFT JOIN brain_relations r ON e.id = r.from_entity OR e.id = r.to_entity
                 GROUP BY e.name HAVING COUNT(r.id) > 2 ORDER BY connection_count DESC LIMIT 10"
            ).fetch_all(pool).await?;
            let bridge_list: Vec<Value> = bridges.iter().map(|(n, c)| serde_json::json!({"entity": n, "connections": c})).collect();
            Ok(serde_json::json!({"metric": "bridges", "bridge_entities": bridge_list}))
        }
        _ => anyhow::bail!("Invalid metric: {metric}"),
    }
}
