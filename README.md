# 🧠 Cuba-Memorys

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, Hebbian learning, and anti-hallucination grounding.

12 tools. 1 dependency. Zero manual setup.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this by giving them:

- **A knowledge graph** — Entities, observations, and relations that persist across sessions
- **Error memory** — Never repeat the same mistake twice (anti-repetition guard)
- **Hebbian learning** — Memories that strengthen with use and fade with time (like a brain)
- **Anti-hallucination** — Verify claims against stored knowledge before answering

| Feature | Cuba-Memorys | Basic Memory MCPs |
|---------|:------------:|:-----------------:|
| Knowledge graph with relations | ✅ | ❌ |
| Hebbian learning (Oja's rule) | ✅ | ❌ |
| Ebbinghaus forgetting curve | ✅ | ❌ |
| Fuzzy search (typo-tolerant) | ✅ | ❌ |
| Error pattern detection | ✅ | ❌ |
| Anti-hallucination grounding | ✅ | ❌ |
| Concept drift detection (χ²) | ✅ | ❌ |
| Spreading activation | ✅ | ❌ |
| Single dependency (asyncpg) | ✅ | ❌ |
| Auto-provisions its own DB | ✅ | ❌ |

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Docker** (for PostgreSQL)

### 2. Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USER/cuba-memorys.git
cd cuba-memorys

# Start PostgreSQL (auto-creates database + schema)
docker compose up -d

# Install the package
pip install -e .
```

### 3. Configure your AI editor

Add to your MCP configuration (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba_memorys_launcher.sh",
      "disabled": false
    }
  }
}
```

Or run directly:

```bash
DATABASE_URL="postgresql://cuba:memorys2026@127.0.0.1:5488/brain" python -m cuba_memorys
```

The server auto-creates the `brain` database and all tables on first run. Zero manual SQL.

---

## The 12 Tools

### Knowledge Graph

| Tool | Description |
|------|------------|
| `brain_entity` | Create, read, update, delete knowledge entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Accessing an entity triggers spreading activation on its neighbors. |
| `brain_observe` | Attach observations to entities. Types: `fact`, `decision`, `lesson`, `preference`, `context`, `tool_usage`. Provenance tracking via `source` (agent/user/error_detection). |
| `brain_relate` | Connect entities with typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). Supports bidirectional edges. **Traverse** walks the graph with strength decay 0.9^(d-1). **Infer** discovers transitive connections. |

### Search & Verification

| Tool | Description |
|------|------------|
| `brain_search` | 4-mode search: **hybrid** (full-text 35% + fuzzy 30% + importance 25% + freshness 10%), **keyword** (PostgreSQL tsvector), **fuzzy** (pg_trgm — tolerates typos), **verify** (anti-hallucination grounding). |

### Error Memory

| Tool | Description |
|------|------------|
| `brain_error_report` | Report errors with context (file, function, stack trace). Auto-detects patterns when ≥3 similar errors exist. Boosts synapse weight of similar errors via Hebbian learning. |
| `brain_error_solve` | Mark an error as resolved with its solution. Cross-references similar unresolved errors so the agent can suggest fixes. |
| `brain_error_query` | Search error memory with hybrid scoring. Use `proposed_action` for the **anti-repetition guard**: warns if a similar approach previously failed. |

### Sessions & Decisions

| Tool | Description |
|------|------------|
| `brain_session` | Track working sessions with goals and outcomes (success/partial/failed/abandoned). |
| `brain_decision` | Record architecture and design decisions with context, alternatives considered, chosen option, and rationale. Query past decisions before making new ones. |

### Memory Maintenance

| Tool | Description |
|------|------------|
| `brain_consolidate` | **decay** — Ebbinghaus 30-day half-life. **prune** — Remove low-importance memories. **merge** — Deduplicate similar entries. **summarize** — Compress observations into a summary. **stats** — Database overview. |
| `brain_feedback` | RLHF loop: **positive** boosts importance (Oja's rule), **negative** decreases it, **correct** updates content. |
| `brain_analytics` | **summary** — Entity/observation/error counts. **health** — Staleness, DB size, unused entities. **drift** — Chi-squared concept drift detection on error types. |

---

## Mathematical Foundations

Cuba-Memorys is built on peer-reviewed algorithms, not ad-hoc heuristics:

### Oja's Rule (1982) — Hebbian Learning

```
Positive: Δw = η · (1 - w²)     → converges to 1.0, cannot explode
Negative: Δw = η · (1 + w²)     → converges to 0.01 (floor)
```

Where `η = 0.05` (learning rate). The `w²` term provides natural saturation — memories can't have infinite importance.

### Ebbinghaus Forgetting Curve (1885)

```
R(t) = R₀ · e^(-λt)     where λ = ln(2)/30 ≈ 0.0231
```

Memories lose 50% importance after 30 days without access. Accessing a memory resets its timer.

### Spreading Activation — Collins & Loftus (1975)

When entity X is accessed, its graph neighbors receive a small importance boost (0.6% per hop, decaying 30% per level). This models associative memory.

### Synapse Weight — Hebbian Error Learning

```
Δw = 0.1 · (1 - w/w_max)     → saturates at w_max = 5.0
```

Similar errors strengthen each other's synapse weights. Frequently co-occurring errors become strongly linked.

### Graph Traversal — Transitive Strength Decay

```
strength(depth) = 0.9^(depth-1)
```

Direct neighbors have strength 1.0. Two hops: 0.9. Three hops: 0.81. Maximum depth: 5 hops.

### Concept Drift — Chi-Squared Test

```
χ² = Σ (observed - expected)² / expected
```

Detects when the distribution of error types changes significantly (critical value: 9.49, df=4, α=0.05).

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml          # Dedicated PostgreSQL (port 5488)
├── pyproject.toml              # Package metadata
├── README.md
└── src/cuba_memorys/
    ├── __init__.py
    ├── __main__.py             # Entry point
    ├── server.py               # 12 MCP tool handlers (~1420 LOC)
    ├── db.py                   # asyncpg pool + auto-DB creation
    ├── schema.sql              # 5 tables, 15 indexes, pg_trgm
    ├── hebbian.py              # Pure math: Oja, Ebbinghaus, spreading
    └── search.py               # LRU cache for search results
```

### Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | Knowledge graph nodes | tsvector + pg_trgm indexes, importance ∈ [0,1] |
| `brain_observations` | Facts attached to entities | 6 types, provenance tracking, FK cascade |
| `brain_relations` | Graph edges | 5 types, bidirectional, unique constraint |
| `brain_errors` | Error memory | JSONB context, synapse weight, solution tracking |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking |

### Search Pipeline

Cuba-Memorys uses **Reciprocal Rank Fusion (RRF)** to combine 4 signals:

1. **Full-text search** (35%) — PostgreSQL `tsvector` with `ts_rank`
2. **Fuzzy matching** (30%) — `pg_trgm` similarity (tolerates typos)
3. **Importance** (25%) — Hebbian-weighted relevance
4. **Freshness** (10%) — Recent memories score higher

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |

### Docker Compose

The included `docker-compose.yml` runs a dedicated PostgreSQL 18 Alpine instance:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always (auto-starts on boot)
- **Healthcheck**: `pg_isready` every 10s

### Customizing the Launcher

Edit `cuba_memorys_launcher.sh` to change connection parameters:

```bash
DB_PORT="5488"      # Change if 5488 is taken
DB_USER="cuba"
DB_PASS="memorys2026"
DB_NAME="brain"
```

---

## Verification

Cuba-Memorys has been tested with a NEMESIS protocol (3-level) test suite:

```
216 tests passed, 0 failed — ZERO DEFECTS

🟢 Normal   — Happy path for all 12 tools, 6 entity types, 6 obs types
🟡 Pessimist — Empty inputs, nonexistent entities, invalid UUIDs, duplicates
🔴 Extreme  — SQL injection, XSS payloads, 10KB content, 10 concurrent creates,
               Unicode/emoji, mathematical precision validation (10⁻¹² tolerance)
```

### Performance

| Operation | Avg latency (20 iters) |
|-----------|:----------------------:|
| Hybrid search | < 5ms |
| Analytics | < 2.5ms |
| Entity CRUD | < 1ms |

---

## How It Works in Practice

### 1. The agent learns from your project

```
Agent: I learned that FastAPI endpoints must use async def with response_model.
→ brain_entity(create, "FastAPI", technology)
→ brain_observe(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: I got IntegrityError: duplicate key on numero_parte.
→ brain_error_report("IntegrityError", "duplicate key on numero_parte")
→ Similar error found! Solution: "Add SELECT EXISTS before INSERT with FOR UPDATE"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify this claim before responding...
→ brain_search("FastAPI uses Django ORM", mode="verify")
→ confidence: "unknown" — not grounded in stored knowledge
```

### 4. Memories fade naturally

```
Day 0:  importance = 1.0 (just learned)
Day 30: importance = 0.5 (Ebbinghaus half-life)
Day 60: importance = 0.25
Access: importance resets and Oja-boosts back up
```

---

## License

MIT

---

## Credits

Built with 🧠 by [@lENADRO1910](https://github.com/lENADRO1910)

Mathematical foundations: Oja (1982), Ebbinghaus (1885), Collins & Loftus (1975)
