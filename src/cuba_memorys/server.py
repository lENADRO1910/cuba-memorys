"""Cuba-Memorys MCP Server: 12-tool Knowledge Graph with Hebbian learning.

JSON-RPC over stdio. Implements the Model Context Protocol (MCP).
Requires: DATABASE_URL env var pointing to PostgreSQL >=12 with pg_trgm.

Tools:
  Knowledge Graph: brain_entity, brain_observe, brain_relate, brain_search
  Error Memory:    brain_error_report, brain_error_solve, brain_error_query
  Sessions:        brain_session, brain_decision
  Intelligence:    brain_consolidate, brain_feedback, brain_analytics
"""

import asyncio
import json
import sys
from typing import Any

from cuba_memorys import db, search
from cuba_memorys.hebbian import (
    oja_negative,
    oja_positive,
    synapse_weight_boost,
)

TOOL_DEFINITIONS = [
    {
        "name": "brain_entity",
        "description": (
            "Manage knowledge graph entities. "
            "Spreading activation: accessing an entity boosts its neighbors' importance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "get"],
                    "description": "Operation to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Entity name (unique identifier)",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Type: concept, project, technology, person, pattern, config",
                },
                "new_name": {
                    "type": "string",
                    "description": "New name for update action",
                },
            },
            "required": ["action", "name"],
        },
    },
    {
        "name": "brain_observe",
        "description": (
            "Add, delete, or list observations attached to entities. "
            "Supports provenance tracking via source parameter."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "delete", "list"],
                    "description": "Operation to perform",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity to attach observation to",
                },
                "content": {
                    "type": "string",
                    "description": "Observation text",
                },
                "observation_type": {
                    "type": "string",
                    "enum": [
                        "fact", "decision", "lesson", "preference",
                        "context", "tool_usage",
                    ],
                    "description": "Type of observation",
                },
                "source": {
                    "type": "string",
                    "enum": ["agent", "user", "error_detection"],
                    "description": "Who/what created this observation",
                },
            },
            "required": ["action", "entity_name"],
        },
    },
    {
        "name": "brain_relate",
        "description": (
            "Manage knowledge graph relations. "
            "Supports transitive inference with cycle detection (max 5 hops, "
            "strength decay 0.9^depth)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "delete", "traverse", "infer"],
                    "description": "Operation to perform",
                },
                "from_entity": {
                    "type": "string",
                    "description": "Source entity name",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Target entity name",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Relation: uses, causes, implements, depends_on, related_to",
                },
                "bidirectional": {
                    "type": "boolean",
                    "description": "If true, relation goes both ways",
                },
                "start_entity": {
                    "type": "string",
                    "description": "Start point for traverse/infer",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max hops for traverse/infer (default 3, max 5)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "brain_search",
        "description": (
            "Hybrid search across knowledge graph using 4 signals: "
            "full-text (35%), fuzzy (30%), importance (25%), freshness (10%). "
            "Mode 'verify' checks anti-hallucination grounding."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "keyword", "fuzzy", "graph", "verify"],
                    "description": "Search mode (default: hybrid). 'verify' checks if claim is grounded.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "entities", "observations", "errors"],
                    "description": "Where to search (default: all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_error_report",
        "description": (
            "Report an error with context. Auto-detects patterns (>=3 similar errors). "
            "Boosts synapse weight of similar existing errors (Hebbian)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "description": "Error category: TypeError, ConnectionError, etc.",
                },
                "error_message": {
                    "type": "string",
                    "description": "Full error message",
                },
                "context": {
                    "type": "object",
                    "description": "Context: {file, function, stack_trace, line}",
                },
                "project": {
                    "type": "string",
                    "description": "Project name (default: 'default')",
                },
            },
            "required": ["error_type", "error_message"],
        },
    },
    {
        "name": "brain_error_solve",
        "description": (
            "Mark an error as resolved with solution. "
            "Cross-references similar unresolved errors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_id": {
                    "type": "string",
                    "description": "UUID of the error to solve",
                },
                "solution": {
                    "type": "string",
                    "description": "Solution that fixed the error",
                },
            },
            "required": ["error_id", "solution"],
        },
    },
    {
        "name": "brain_error_query",
        "description": (
            "Search error memory with hybrid scoring. "
            "Use 'proposed_action' for anti-repetition guard: warns if a "
            "similar approach previously failed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text for errors",
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project",
                },
                "resolved_only": {
                    "type": "boolean",
                    "description": "Only return errors with solutions",
                },
                "proposed_action": {
                    "type": "string",
                    "description": "Anti-repetition: describe what you plan to do. "
                    "Returns warning if similar approach failed before.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_session",
        "description": "Track working sessions with goals and outcomes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "end", "list", "current"],
                    "description": "Session action",
                },
                "name": {
                    "type": "string",
                    "description": "Session name (for start)",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Session goals (for start)",
                },
                "summary": {
                    "type": "string",
                    "description": "What was accomplished (for end)",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial", "failed", "abandoned"],
                    "description": "Session outcome (for end)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "brain_decision",
        "description": "Record and query architecture/design decisions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["record", "query", "list"],
                    "description": "Decision action",
                },
                "title": {
                    "type": "string",
                    "description": "Decision title (for record)",
                },
                "context": {
                    "type": "string",
                    "description": "Why this decision was needed",
                },
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options considered",
                },
                "chosen": {
                    "type": "string",
                    "description": "Option chosen",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this option was chosen",
                },
                "query": {
                    "type": "string",
                    "description": "Search text (for query action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "brain_consolidate",
        "description": (
            "Memory maintenance: decay (Ebbinghaus 30-day half-life), "
            "prune (remove low-importance), merge (deduplicate similar), "
            "summarize (compress observations for an entity), stats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["decay", "prune", "merge", "summarize", "stats"],
                    "description": "Consolidation action",
                },
                "threshold": {
                    "type": "number",
                    "description": "Importance threshold for prune (default 0.1)",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Similarity threshold for merge (default 0.8)",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity to summarize (for summarize action)",
                },
                "compressed_summary": {
                    "type": "string",
                    "description": "Compressed text replacing observations (for summarize)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "brain_feedback",
        "description": (
            "RLHF feedback loop: reinforce useful memories (Oja's rule). "
            "Positive boosts importance, negative decreases, correct updates content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["positive", "negative", "correct"],
                    "description": "Feedback type",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Target entity",
                },
                "observation_id": {
                    "type": "string",
                    "description": "Target observation UUID",
                },
                "correction": {
                    "type": "string",
                    "description": "New content (for correct action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "brain_analytics",
        "description": (
            "Knowledge graph analytics: summary (counts), "
            "health (staleness, DB size), "
            "drift (chi-squared concept drift detection on error types)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["summary", "health", "drift"],
                    "description": "Metric to compute",
                },
            },
            "required": ["metric"],
        },
    },
]


# ─── Tool Handlers ───────────────────────────────────────────────────

async def handle_brain_entity(args: dict[str, Any]) -> str:
    """Handle brain_entity tool calls."""
    action = args["action"]
    name = args.get("name", "")

    if action == "create":
        entity_type = args.get("entity_type", "concept")
        row = await db.fetchrow(
            "INSERT INTO brain_entities (name, entity_type) "
            "VALUES ($1, $2) "
            "ON CONFLICT (name) DO UPDATE SET updated_at = NOW() "
            "RETURNING id, name, entity_type, importance",
            name, entity_type,
        )
        return db.serialize({"action": "created", "entity": row})

    if action == "get":
        row = await db.fetchrow(
            "SELECT id, name, entity_type, importance, access_count, "
            "created_at, updated_at FROM brain_entities WHERE name = $1",
            name,
        )
        if not row:
            return json.dumps({"error": f"Entity '{name}' not found"})

        entity_id = row["id"]

        # Spreading activation: boost self
        await db.execute(
            "UPDATE brain_entities SET access_count = access_count + 1, "
            "importance = LEAST(1.0, importance + 0.02), "
            "updated_at = NOW() WHERE id = $1",
            entity_id,
        )
        # FIX BUG-6: Spreading activation boost neighbors
        # asyncpg passes UUID natively — no ::uuid cast needed
        await db.execute(
            "UPDATE brain_entities SET "
            "importance = LEAST(1.0, importance + 0.006) "
            "WHERE id IN ("
            "  SELECT CASE WHEN r.from_entity = $1 THEN r.to_entity "
            "              ELSE r.from_entity END "
            "  FROM brain_relations r "
            "  WHERE r.from_entity = $1 "
            "     OR (r.to_entity = $1 AND r.bidirectional = TRUE)"
            ")",
            entity_id,
        )

        # Get observations
        obs = await db.fetch(
            "SELECT id, content, observation_type, importance, source, "
            "created_at FROM brain_observations "
            "WHERE entity_id = $1 ORDER BY importance DESC",
            entity_id,
        )
        # Get relations
        rels = await db.fetch(
            "SELECT e.name AS target, r.relation_type, r.strength, "
            "r.bidirectional "
            "FROM brain_relations r "
            "JOIN brain_entities e ON r.to_entity = e.id "
            "WHERE r.from_entity = $1 "
            "UNION ALL "
            "SELECT e.name AS target, r.relation_type, r.strength, "
            "r.bidirectional "
            "FROM brain_relations r "
            "JOIN brain_entities e ON r.from_entity = e.id "
            "WHERE r.to_entity = $1 AND r.bidirectional = TRUE",
            entity_id,
        )
        return db.serialize({
            "entity": row,
            "observations": obs,
            "relations": rels,
        })

    if action == "update":
        new_name = args.get("new_name", name)
        await db.execute(
            "UPDATE brain_entities SET name = $1, updated_at = NOW() "
            "WHERE name = $2",
            new_name, name,
        )
        return json.dumps({"action": "updated", "old_name": name, "new_name": new_name})

    if action == "delete":
        await db.execute("DELETE FROM brain_entities WHERE name = $1", name)
        return json.dumps({"action": "deleted", "name": name})

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_observe(args: dict[str, Any]) -> str:
    """Handle brain_observe tool calls."""
    action = args["action"]
    entity_name = args["entity_name"]

    entity = await db.fetchrow(
        "SELECT id FROM brain_entities WHERE name = $1", entity_name,
    )
    if not entity:
        return json.dumps({"error": f"Entity '{entity_name}' not found"})

    entity_id = entity["id"]

    if action == "add":
        content = args.get("content", "")
        obs_type = args.get("observation_type", "fact")
        source = args.get("source", "agent")
        row = await db.fetchrow(
            "INSERT INTO brain_observations "
            "(entity_id, content, observation_type, source) "
            "VALUES ($1, $2, $3, $4) RETURNING id, content, observation_type",
            entity_id, content, obs_type, source,
        )
        search.cache_clear()
        return db.serialize({"action": "added", "observation": row})

    if action == "delete":
        content = args.get("content", "")
        await db.execute(
            "DELETE FROM brain_observations "
            "WHERE entity_id = $1 AND content = $2",
            entity_id, content,
        )
        search.cache_clear()
        return json.dumps({"action": "deleted", "entity": entity_name})

    if action == "list":
        obs = await db.fetch(
            "SELECT id, content, observation_type, importance, source, "
            "access_count, last_accessed, created_at "
            "FROM brain_observations WHERE entity_id = $1 "
            "ORDER BY importance DESC",
            entity_id,
        )
        return db.serialize({"entity": entity_name, "observations": obs})

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_relate(args: dict[str, Any]) -> str:
    """Handle brain_relate tool calls."""
    action = args["action"]

    if action == "create":
        from_name = args["from_entity"]
        to_name = args["to_entity"]
        rel_type = args.get("relation_type", "related_to")
        bidir = args.get("bidirectional", False)

        from_e = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1", from_name,
        )
        to_e = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1", to_name,
        )
        if not from_e or not to_e:
            return json.dumps({"error": "One or both entities not found"})

        await db.execute(
            "INSERT INTO brain_relations "
            "(from_entity, to_entity, relation_type, bidirectional) "
            "VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (from_entity, to_entity, relation_type) DO NOTHING",
            from_e["id"], to_e["id"], rel_type, bidir,
        )
        return json.dumps({
            "action": "created",
            "from": from_name, "to": to_name,
            "type": rel_type, "bidirectional": bidir,
        })

    if action == "delete":
        from_name = args["from_entity"]
        to_name = args["to_entity"]
        rel_type = args.get("relation_type", "")

        from_e = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1", from_name,
        )
        to_e = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1", to_name,
        )
        if not from_e or not to_e:
            return json.dumps({"error": "One or both entities not found"})

        if rel_type:
            await db.execute(
                "DELETE FROM brain_relations "
                "WHERE from_entity = $1 AND to_entity = $2 "
                "AND relation_type = $3",
                from_e["id"], to_e["id"], rel_type,
            )
        else:
            await db.execute(
                "DELETE FROM brain_relations "
                "WHERE from_entity = $1 AND to_entity = $2",
                from_e["id"], to_e["id"],
            )
        return json.dumps({
            "action": "deleted", "from": from_name, "to": to_name,
        })

    if action in ("traverse", "infer"):
        start_name = args.get("start_entity", "")
        max_depth = min(args.get("max_depth", 3), 5)

        start_e = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1", start_name,
        )
        if not start_e:
            return json.dumps({"error": f"Entity '{start_name}' not found"})

        rows = await db.fetch(
            "WITH RECURSIVE inference AS ("
            "  SELECT r.to_entity, r.strength, 1 AS depth, "
            "    ARRAY[r.from_entity] AS path "
            "  FROM brain_relations r WHERE r.from_entity = $1 "
            "  UNION ALL "
            "  SELECT r.to_entity, "
            "    i.strength * r.strength * 0.9, "
            "    i.depth + 1, "
            "    i.path || r.from_entity "
            "  FROM brain_relations r "
            "  JOIN inference i ON r.from_entity = i.to_entity "
            "  WHERE i.depth < $2 "
            "    AND r.from_entity != ALL(i.path) "
            "    AND i.strength * r.strength > 0.1"
            ") "
            "SELECT e.name, e.entity_type, inf.strength, inf.depth "
            "FROM inference inf "
            "JOIN brain_entities e ON inf.to_entity = e.id "
            "ORDER BY inf.strength DESC "
            "LIMIT 20",
            start_e["id"], max_depth,
        )
        return db.serialize({
            "action": action, "start": start_name,
            "max_depth": max_depth, "results": rows,
        })

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_search(args: dict[str, Any]) -> str:
    """Handle brain_search tool calls."""
    query = args["query"]
    mode = args.get("mode", "hybrid")
    scope = args.get("scope", "all")
    limit = min(args.get("limit", 10), 50)

    # Anti-hallucination verify mode
    if mode == "verify":
        row = await db.fetchrow(search.VERIFY_SQL, query)
        return db.serialize({
            "mode": "verify",
            "query": query,
            "confidence": row["confidence"] if row else "unknown",
            "closest_match": row["closest_match"] if row else None,
        })

    # LRU cache check
    cached = search.cache_get(query, mode, scope, limit)
    if cached is not None:
        return db.serialize({"cached": True, "results": cached})

    results: list[dict[str, Any]] = []

    if scope in ("all", "entities"):
        rows = await db.fetch(search.SEARCH_ENTITIES_SQL, query, limit)
        for r in rows:
            r["_type"] = "entity"
        results.extend(rows)

    if scope in ("all", "observations"):
        rows = await db.fetch(search.SEARCH_OBSERVATIONS_SQL, query, limit)
        for r in rows:
            r["_type"] = "observation"
        results.extend(rows)

    if scope in ("all", "errors"):
        rows = await db.fetch(search.SEARCH_ERRORS_SQL, query, limit)
        for r in rows:
            r["_type"] = "error"
        results.extend(rows)

    # Sort all results by score and limit
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    results = results[:limit]

    search.cache_set(query, mode, scope, limit, results)
    return db.serialize({"results": results})


async def handle_brain_error_report(args: dict[str, Any]) -> str:
    """Handle brain_error_report tool calls."""
    error_type = args["error_type"]
    error_message = args["error_message"]
    context = json.dumps(args.get("context", {}))
    project = args.get("project", "default")

    # Insert new error
    row = await db.fetchrow(
        "INSERT INTO brain_errors "
        "(error_type, error_message, context, project) "
        "VALUES ($1, $2, $3::jsonb, $4) "
        "RETURNING id, error_type, error_message, created_at",
        error_type, error_message, context, project,
    )
    if not row:
        return json.dumps({"error": "Failed to insert error record"})

    # Find similar errors (pattern detection)
    similar = await db.fetch(
        "SELECT id, error_message, solution, resolved, synapse_weight "
        "FROM brain_errors "
        "WHERE id != $1 AND similarity(error_message, $2) > 0.4 "
        "ORDER BY similarity(error_message, $2) DESC LIMIT 5",
        row["id"], error_message,
    )

    # Boost synapse weight of similar errors (Hebbian)
    for s in similar:
        new_weight = synapse_weight_boost(s["synapse_weight"])
        await db.execute(
            "UPDATE brain_errors SET synapse_weight = $1 WHERE id = $2",
            new_weight, s["id"],
        )

    is_pattern = len(similar) >= 2  # 3+ total including new one
    search.cache_clear()

    return db.serialize({
        "action": "reported",
        "error": row,
        "similar_count": len(similar),
        "is_pattern": is_pattern,
        "similar_errors": similar,
    })


async def handle_brain_error_solve(args: dict[str, Any]) -> str:
    """Handle brain_error_solve tool calls."""
    error_id = args["error_id"]
    solution = args["solution"]

    await db.execute(
        "UPDATE brain_errors SET solution = $1, resolved = TRUE, "
        "resolved_at = NOW() WHERE id = $2::uuid",
        solution, error_id,
    )

    # Cross-reference: find similar unresolved errors
    error = await db.fetchrow(
        "SELECT error_message FROM brain_errors WHERE id = $1::uuid",
        error_id,
    )
    candidates: list[dict[str, Any]] = []
    if error:
        candidates = await db.fetch(
            "SELECT id, error_message, synapse_weight "
            "FROM brain_errors "
            "WHERE resolved = FALSE AND id != $1::uuid "
            "AND similarity(error_message, $2) > 0.5 "
            "ORDER BY similarity(error_message, $2) DESC LIMIT 5",
            error_id, error["error_message"],
        )

    search.cache_clear()
    return db.serialize({
        "action": "solved",
        "error_id": error_id,
        "solution": solution,
        "similar_unresolved": candidates,
    })


async def handle_brain_error_query(args: dict[str, Any]) -> str:
    """Handle brain_error_query tool calls.

    FIX BUG-2: Replaced fragile string manipulation with proper
    parameterized SQL construction. Each combination of filters
    gets its own SQL query to avoid parameter count mismatches.
    """
    query_text = args["query"]
    project = args.get("project")
    resolved_only = args.get("resolved_only", False)
    proposed_action = args.get("proposed_action")

    # Anti-repetition guard
    if proposed_action:
        failed = await db.fetch(
            "SELECT id, error_type, error_message, context, solution "
            "FROM brain_errors "
            "WHERE resolved = FALSE "
            "AND similarity(error_message, $1) > 0.5 "
            "ORDER BY similarity(error_message, $1) DESC LIMIT 3",
            proposed_action,
        )
        if failed:
            return db.serialize({
                "warning": "ANTI-REPETITION: Similar approaches have failed before",
                "proposed_action": proposed_action,
                "failed_attempts": failed,
            })

    # FIX BUG-2: Build SQL with proper parameterization
    # Each filter combination uses explicit SQL (no string replace)
    if project and resolved_only:
        rows = await db.fetch(
            "SELECT id, error_type, error_message, solution, resolved, "
            "synapse_weight, project, created_at, "
            "(0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) + "
            " 0.30 * similarity(error_message, $1) + "
            " 0.30 * (synapse_weight / GREATEST("
            "   (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))"
            ") AS score "
            "FROM brain_errors "
            "WHERE (search_vector @@ plainto_tsquery('simple', $1) "
            "   OR similarity(error_message, $1) > 0.3) "
            "AND project = $2 AND resolved = TRUE "
            "ORDER BY score DESC LIMIT 10",
            query_text, project,
        )
    elif project:
        rows = await db.fetch(
            "SELECT id, error_type, error_message, solution, resolved, "
            "synapse_weight, project, created_at, "
            "(0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) + "
            " 0.30 * similarity(error_message, $1) + "
            " 0.30 * (synapse_weight / GREATEST("
            "   (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))"
            ") AS score "
            "FROM brain_errors "
            "WHERE (search_vector @@ plainto_tsquery('simple', $1) "
            "   OR similarity(error_message, $1) > 0.3) "
            "AND project = $2 "
            "ORDER BY score DESC LIMIT 10",
            query_text, project,
        )
    elif resolved_only:
        rows = await db.fetch(
            "SELECT id, error_type, error_message, solution, resolved, "
            "synapse_weight, project, created_at, "
            "(0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) + "
            " 0.30 * similarity(error_message, $1) + "
            " 0.30 * (synapse_weight / GREATEST("
            "   (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))"
            ") AS score "
            "FROM brain_errors "
            "WHERE (search_vector @@ plainto_tsquery('simple', $1) "
            "   OR similarity(error_message, $1) > 0.3) "
            "AND resolved = TRUE "
            "ORDER BY score DESC LIMIT 10",
            query_text,
        )
    else:
        rows = await db.fetch(search.SEARCH_ERRORS_SQL, query_text, 10)

    return db.serialize({"results": rows})


async def handle_brain_session(args: dict[str, Any]) -> str:
    """Handle brain_session tool calls."""
    action = args["action"]

    if action == "start":
        name = args.get("name", "Unnamed Session")
        goals = json.dumps(args.get("goals", []))
        row = await db.fetchrow(
            "INSERT INTO brain_sessions (session_name, goals) "
            "VALUES ($1, $2::jsonb) RETURNING id, session_name, started_at",
            name, goals,
        )
        return db.serialize({"action": "started", "session": row})

    if action == "end":
        summary = args.get("summary", "")
        outcome = args.get("outcome", "success")
        # FIX BUG-1: PostgreSQL does NOT support ORDER BY/LIMIT in UPDATE.
        # Use subquery with ctid or id to select the target row.
        row = await db.fetchrow(
            "UPDATE brain_sessions SET summary = $1, outcome = $2, "
            "ended_at = NOW() "
            "WHERE id = ("
            "  SELECT id FROM brain_sessions "
            "  WHERE ended_at IS NULL "
            "  ORDER BY started_at DESC LIMIT 1"
            ") "
            "RETURNING id, session_name, outcome, started_at, ended_at",
            summary, outcome,
        )
        if not row:
            return json.dumps({"error": "No active session to end"})
        return db.serialize({"action": "ended", "session": row})

    if action == "current":
        row = await db.fetchrow(
            "SELECT id, session_name, goals, started_at "
            "FROM brain_sessions WHERE ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
        )
        if not row:
            return json.dumps({"status": "no_active_session"})
        return db.serialize({"session": row})

    if action == "list":
        rows = await db.fetch(
            "SELECT id, session_name, outcome, started_at, ended_at "
            "FROM brain_sessions ORDER BY started_at DESC LIMIT 20",
        )
        return db.serialize({"sessions": rows})

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_decision(args: dict[str, Any]) -> str:
    """Handle brain_decision tool calls."""
    action = args["action"]

    if action == "record":
        # Store as observation on a special _decisions entity
        entity = await db.fetchrow(
            "INSERT INTO brain_entities (name, entity_type) "
            "VALUES ('_decisions', 'system') "
            "ON CONFLICT (name) DO UPDATE SET updated_at = NOW() "
            "RETURNING id",
        )
        if not entity:
            return json.dumps({"error": "Failed to create _decisions entity"})

        decision = {
            "title": args.get("title", ""),
            "context": args.get("context", ""),
            "alternatives": args.get("alternatives", []),
            "chosen": args.get("chosen", ""),
            "rationale": args.get("rationale", ""),
        }
        row = await db.fetchrow(
            "INSERT INTO brain_observations "
            "(entity_id, content, observation_type, source) "
            "VALUES ($1, $2, 'decision', 'agent') "
            "RETURNING id, created_at",
            entity["id"], json.dumps(decision),
        )
        if not row:
            return json.dumps({"error": "Failed to insert decision"})

        search.cache_clear()
        return db.serialize({"action": "recorded", "decision": decision, "id": row["id"]})

    if action == "query":
        query_text = args.get("query", "")
        rows = await db.fetch(
            "SELECT id, content, importance, created_at "
            "FROM brain_observations "
            "WHERE observation_type = 'decision' "
            "AND (search_vector @@ plainto_tsquery('simple', $1) "
            "     OR similarity(content, $1) > 0.3) "
            "ORDER BY importance DESC LIMIT 10",
            query_text,
        )
        decisions = []
        for r in rows:
            try:
                d = json.loads(r["content"])
                d["id"] = str(r["id"])
                d["importance"] = r["importance"]
                decisions.append(d)
            except json.JSONDecodeError:
                decisions.append({"raw": r["content"], "id": str(r["id"])})
        return db.serialize({"decisions": decisions})

    if action == "list":
        rows = await db.fetch(
            "SELECT id, content, importance, created_at "
            "FROM brain_observations "
            "WHERE observation_type = 'decision' "
            "ORDER BY created_at DESC LIMIT 20",
        )
        decisions = []
        for r in rows:
            try:
                d = json.loads(r["content"])
                d["id"] = str(r["id"])
                decisions.append(d)
            except json.JSONDecodeError:
                decisions.append({"raw": r["content"], "id": str(r["id"])})
        return db.serialize({"decisions": decisions})

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_consolidate(args: dict[str, Any]) -> str:
    """Handle brain_consolidate tool calls."""
    action = args["action"]

    if action == "decay":
        # Ebbinghaus decay: importance *= e^(-0.0231 * days_since_access)
        result = await db.execute(
            "UPDATE brain_observations SET "
            "importance = GREATEST(0.01, "
            "  importance * EXP(-0.0231 * "
            "    EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0)) "
            "WHERE last_accessed < NOW() - INTERVAL '1 day'",
        )
        search.cache_clear()
        return json.dumps({"action": "decay_applied", "status": result})

    if action == "prune":
        threshold = args.get("threshold", 0.1)
        result = await db.execute(
            "DELETE FROM brain_observations "
            "WHERE importance < $1 AND access_count < 2",
            threshold,
        )
        search.cache_clear()
        return json.dumps({"action": "pruned", "threshold": threshold, "status": result})

    if action == "merge":
        sim_threshold = args.get("similarity_threshold", 0.8)
        # Find duplicate observations
        dupes = await db.fetch(
            "SELECT a.id AS id_a, b.id AS id_b, "
            "a.content AS content_a, b.content AS content_b, "
            "similarity(a.content, b.content) AS sim "
            "FROM brain_observations a "
            "JOIN brain_observations b ON a.entity_id = b.entity_id "
            "  AND a.id < b.id "
            "WHERE similarity(a.content, b.content) > $1 "
            "LIMIT 50",
            sim_threshold,
        )
        merged = 0
        for d in dupes:
            # Keep the one with higher importance, delete the other
            await db.execute(
                "UPDATE brain_observations SET "
                "importance = LEAST(1.0, importance + 0.1) "
                "WHERE id = $1",
                d["id_a"],
            )
            await db.execute(
                "DELETE FROM brain_observations WHERE id = $1",
                d["id_b"],
            )
            merged += 1
        search.cache_clear()
        return json.dumps({"action": "merged", "pairs_merged": merged})

    if action == "summarize":
        entity_name = args.get("entity_name", "")
        compressed = args.get("compressed_summary", "")

        entity = await db.fetchrow(
            "SELECT id FROM brain_entities WHERE name = $1",
            entity_name,
        )
        if not entity:
            return json.dumps({"error": f"Entity '{entity_name}' not found"})

        count = await db.fetchval(
            "SELECT COUNT(*) FROM brain_observations WHERE entity_id = $1",
            entity["id"],
        )
        # Delete old observations and insert compressed summary
        await db.execute(
            "DELETE FROM brain_observations WHERE entity_id = $1",
            entity["id"],
        )
        await db.execute(
            "INSERT INTO brain_observations "
            "(entity_id, content, observation_type, importance, source) "
            "VALUES ($1, $2, 'fact', 0.9, 'consolidation')",
            entity["id"], compressed,
        )
        search.cache_clear()
        return json.dumps({
            "action": "summarized",
            "entity": entity_name,
            "observations_replaced": count,
        })

    if action == "stats":
        stats = await db.fetchrow(
            "SELECT "
            "  (SELECT COUNT(*) FROM brain_entities) AS entities, "
            "  (SELECT COUNT(*) FROM brain_observations) AS observations, "
            "  (SELECT COUNT(*) FROM brain_relations) AS relations, "
            "  (SELECT COUNT(*) FROM brain_errors) AS errors, "
            "  (SELECT COUNT(*) FROM brain_errors WHERE resolved) AS errors_resolved, "
            "  (SELECT COUNT(*) FROM brain_sessions) AS sessions, "
            "  (SELECT AVG(importance) FROM brain_observations) AS avg_importance",
        )
        return db.serialize({"stats": stats})

    return json.dumps({"error": f"Unknown action: {action}"})


async def handle_brain_feedback(args: dict[str, Any]) -> str:
    """Handle brain_feedback tool calls."""
    action = args["action"]
    entity_name = args.get("entity_name")
    obs_id = args.get("observation_id")
    correction = args.get("correction")

    if entity_name:
        row = await db.fetchrow(
            "SELECT id, importance FROM brain_entities WHERE name = $1",
            entity_name,
        )
        if not row:
            return json.dumps({"error": f"Entity '{entity_name}' not found"})

        if action == "positive":
            new_imp = oja_positive(row["importance"])
        elif action == "negative":
            new_imp = oja_negative(row["importance"])
        else:
            return json.dumps({"error": "correct requires observation_id"})

        await db.execute(
            "UPDATE brain_entities SET importance = $1, updated_at = NOW() "
            "WHERE id = $2",
            new_imp, row["id"],
        )
        search.cache_clear()
        return json.dumps({
            "action": action,
            "entity": entity_name,
            "old_importance": row["importance"],
            "new_importance": new_imp,
        })

    if obs_id:
        row = await db.fetchrow(
            "SELECT id, importance, content FROM brain_observations "
            "WHERE id = $1::uuid",
            obs_id,
        )
        if not row:
            return json.dumps({"error": f"Observation '{obs_id}' not found"})

        if action == "positive":
            new_imp = oja_positive(row["importance"])
            await db.execute(
                "UPDATE brain_observations SET importance = $1, "
                "access_count = access_count + 1, last_accessed = NOW() "
                "WHERE id = $2::uuid",
                new_imp, obs_id,
            )
        elif action == "negative":
            new_imp = oja_negative(row["importance"])
            await db.execute(
                "UPDATE brain_observations SET importance = $1 "
                "WHERE id = $2::uuid",
                new_imp, obs_id,
            )
        elif action == "correct" and correction:
            new_imp = row["importance"]
            await db.execute(
                "UPDATE brain_observations SET content = $1, "
                "last_accessed = NOW() WHERE id = $2::uuid",
                correction, obs_id,
            )
        else:
            return json.dumps({"error": "correction text required"})

        search.cache_clear()
        return json.dumps({
            "action": action,
            "observation_id": obs_id,
            "old_importance": row["importance"],
            "new_importance": new_imp,
        })

    return json.dumps({"error": "entity_name or observation_id required"})


async def handle_brain_analytics(args: dict[str, Any]) -> str:
    """Handle brain_analytics tool calls."""
    metric = args["metric"]

    if metric == "summary":
        row = await db.fetchrow(
            "SELECT "
            "  (SELECT COUNT(*) FROM brain_entities) AS entities, "
            "  (SELECT COUNT(*) FROM brain_observations) AS observations, "
            "  (SELECT COUNT(*) FROM brain_relations) AS relations, "
            "  (SELECT COUNT(*) FROM brain_errors) AS total_errors, "
            "  (SELECT COUNT(*) FROM brain_errors WHERE resolved) AS resolved_errors, "
            "  (SELECT COUNT(*) FROM brain_sessions) AS sessions",
        )
        return db.serialize({"summary": row})

    if metric == "health":
        row = await db.fetchrow(
            "SELECT "
            "  (SELECT AVG(importance) FROM brain_observations) AS avg_importance, "
            "  (SELECT COUNT(*) FROM brain_observations "
            "   WHERE last_accessed < NOW() - INTERVAL '90 days') AS stale_observations, "
            "  (SELECT COUNT(*) FROM brain_entities "
            "   WHERE access_count = 0) AS unused_entities, "
            "  (SELECT pg_size_pretty(pg_database_size(current_database()))) AS db_size",
        )
        return db.serialize({"health": row})

    if metric == "drift":
        # Chi-squared concept drift detection
        row = await db.fetchrow(
            "WITH recent AS ("
            "  SELECT error_type, COUNT(*)::float AS cnt "
            "  FROM brain_errors WHERE created_at > NOW() - INTERVAL '7 days' "
            "  GROUP BY error_type"
            "), historical AS ("
            "  SELECT error_type, COUNT(*)::float / 4.0 AS expected "
            "  FROM brain_errors "
            "  WHERE created_at BETWEEN NOW() - INTERVAL '37 days' "
            "    AND NOW() - INTERVAL '7 days' "
            "  GROUP BY error_type"
            ") "
            "SELECT "
            "  COALESCE(SUM(POWER(r.cnt - h.expected, 2) / "
            "    NULLIF(h.expected, 0)), 0) AS chi_squared, "
            "  COUNT(*) AS categories "
            "FROM recent r JOIN historical h USING (error_type)",
        )
        chi_sq = float(row["chi_squared"]) if row else 0
        categories = int(row["categories"]) if row else 0
        # Critical value: df=categories-1, α=0.05
        # Approximate: 2*categories for simple threshold
        critical = max(9.49, 2.0 * categories) if categories > 0 else 9.49
        drift_detected = chi_sq > critical

        return json.dumps({
            "chi_squared": round(chi_sq, 4),
            "critical_value": round(critical, 4),
            "categories": categories,
            "drift_detected": drift_detected,
        })

    return json.dumps({"error": f"Unknown metric: {metric}"})


# ─── Tool dispatcher ─────────────────────────────────────────────────

HANDLERS = {
    "brain_entity": handle_brain_entity,
    "brain_observe": handle_brain_observe,
    "brain_relate": handle_brain_relate,
    "brain_search": handle_brain_search,
    "brain_error_report": handle_brain_error_report,
    "brain_error_solve": handle_brain_error_solve,
    "brain_error_query": handle_brain_error_query,
    "brain_session": handle_brain_session,
    "brain_decision": handle_brain_decision,
    "brain_consolidate": handle_brain_consolidate,
    "brain_feedback": handle_brain_feedback,
    "brain_analytics": handle_brain_analytics,
}


# ─── JSON-RPC over stdio ─────────────────────────────────────────────

async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC request.

    Args:
        request: Parsed JSON-RPC request.

    Returns:
        JSON-RPC response dict, or None for notifications.
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "cuba-memorys",
                    "version": "1.0.0",
                },
            },
        }

    if method == "notifications/initialized":
        # FIX BUG-4: Notifications have no id, must NOT send a response.
        # Just init schema silently.
        await db.init_schema()
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    }],
                    "isError": True,
                },
            }

        try:
            result_text = await handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": type(e).__name__,
                            "message": str(e),
                        }),
                    }],
                    "isError": True,
                },
            }

    # Unknown methods that have an id get an error response
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    # Notifications without id get no response
    return None


async def main() -> None:
    """Run the MCP server: read JSON-RPC from stdin, write to stdout.

    Uses sys.stdin.buffer and sys.stdout.buffer directly with
    asyncio protocol for maximum compatibility across Python versions.
    """
    loop = asyncio.get_running_loop()

    # Set up stdin reader
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    # Set up stdout writer — use subprocess protocol pattern
    write_transport, write_protocol = await loop.connect_write_pipe(
        lambda: asyncio.Protocol(), sys.stdout.buffer,
    )

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                request = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            response = await handle_request(request)

            # FIX BUG-4: Only send response for non-notification messages
            if response is not None:
                response_bytes = json.dumps(response).encode("utf-8") + b"\n"
                write_transport.write(response_bytes)

    except (ConnectionError, BrokenPipeError, EOFError):
        pass
    finally:
        await db.close()
        write_transport.close()
