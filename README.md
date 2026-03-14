# 🧠 Cuba-Memorys

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue?logo=python&logoColor=white)](https://python.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/LeandroPG19/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, Hebbian learning with BCM metaplasticity, FSRS-6 adaptive decay, GraphRAG enrichment, and anti-hallucination grounding.

12 tools with Cuban soul. Zero manual setup. Mathematically rigorous.

**v2.0.0** — Complete Rust rewrite. Sub-millisecond handlers, 7.6MB binary, FSRS-6, BCM metaplasticity, Brandes centrality, Shannon entropy analytics, Dual-Strength memory, Leiden communities.

> [!IMPORTANT]
> **v2.0 is written in Rust** — the primary implementation. The Python version (v1.6.0) remains in `src/` as legacy reference.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this by giving them:

- **A knowledge graph** — Entities, observations, and relations that persist across sessions
- **Error memory** — Never repeat the same mistake twice (anti-repetition guard)
- **FSRS-6 adaptive spaced repetition** — Memories strengthen with use and fade adaptively (FSRS-6, 21 parameters, per-entity adaptive decay w20)
- **Anti-hallucination** — Verify claims against stored knowledge with graduated confidence + source diversity scoring
- **Semantic search** — 4-signal RRF fusion (TF-IDF + pg_trgm + full-text + optional pgvector HNSW)
- **KG-neighbor query expansion** — Auto-expands low-recall searches via graph neighbors (PRF + Zep/Graphiti)
- **GraphRAG** — Top results enriched with degree-1 graph neighbors for topological context
- **REM Sleep** — Autonomous background consolidation (FSRS decay + prune + PageRank) after 15min idle
- **Graph intelligence** — Personalized PageRank, Louvain community detection with summaries, betweenness centrality
- **Embedding cache** — LRU cache eliminates redundant ONNX inference (~84% reduction)
- **Session-aware search** — Results matching active session goals boosted by 15%

| Feature | Cuba-Memorys | Basic Memory MCPs |
| ------- | :----------: | :---------------: |
| Knowledge graph with relations | ✅ | ❌ |
| Hebbian learning (Oja's rule) | ✅ | ❌ |
| FSRS-6 adaptive spaced repetition | ✅ | ❌ |
| **Entropy-routed RRF fusion search** | ✅ | ❌ |
| **KG-neighbor query expansion (V9)** | ✅ | ❌ |
| **Embedding LRU cache (V10)** | ✅ | ❌ |
| **GraphRAG topological enrichment** | ✅ | ❌ |
| **REM Sleep autonomous consolidation** | ✅ | ❌ |
| **Community summaries (V12)** | ✅ | ❌ |
| **Source triangulation (V3)** | ✅ | ❌ |
| **Session-aware search (V13)** | ✅ | ❌ |
| **Batch access tracking (V14)** | ✅ | ❌ |
| **Token-budget truncation (V1)** | ✅ | ❌ |
| **Post-fusion deduplication (V2)** | ✅ | ❌ |
| **Adaptive confidence weights (V4)** | ✅ | ❌ |
| **Information density gating (V5)** | ✅ | ❌ |
| **Session-aware FSRS decay (V6)** | ✅ | ❌ |
| **Conditional pgvector + HNSW** | ✅ | ❌ |
| **Modular architecture (CC avg A)** | ✅ | ❌ |
| Optional BGE embeddings (ONNX) | ✅ | ❌ |
| Contradiction detection | ✅ | ❌ |
| Graduated confidence scoring | ✅ | ❌ |
| Personalized PageRank | ✅ | ❌ |
| Louvain community detection + summaries | ✅ | ❌ |
| Betweenness centrality (bridges) | ✅ | ❌ |
| Shannon entropy (knowledge diversity) | ✅ | ❌ |
| Chi-squared concept drift detection | ✅ | ❌ |
| Error pattern detection + MTTR | ✅ | ❌ |
| Entity duplicate detection (SQL similarity) | ✅ | ❌ |
| Observation versioning (audit trail) | ✅ | ❌ |
| Temporal validity (valid_from/valid_until) | ✅ | ❌ |
| Write-time dedup gate | ✅ | ❌ |
| Auto-supersede contradictions | ✅ | ❌ |
| Full JSON export/backup (bounded) | ✅ | ❌ |
| Fuzzy search (typo-tolerant) | ✅ | ❌ |
| Spreading activation | ✅ | ❌ |
| Batch observations (10x fewer calls) | ✅ | ❌ |
| Entity type validation | ✅ | ❌ |
| Graceful shutdown (SIGTERM/SIGINT) | ✅ | ❌ |
| Auto-provisions its own DB | ✅ | ❌ |

---

## Quick Start

### Rust (Recommended — v2.0)

```bash
git clone https://github.com/LeandroPG19/cuba-memorys.git
cd cuba-memorys

# Start PostgreSQL
docker compose up -d

# Build Rust binary
cd rust
cargo build --release
```

Configure your AI editor:

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba-memorys/rust/target/release/cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://cuba:memorys2026@127.0.0.1:5488/brain"
      }
    }
  }
}
```

The server auto-creates the `brain` database and all tables on first run.

### Python (Legacy — v1.6.0)

```bash
pip install -e .
DATABASE_URL="postgresql://cuba:memorys2026@127.0.0.1:5488/brain" python -m cuba_memorys
```

---

## 🇨🇺 Las 12 Herramientas

Every tool is named after Cuban culture — memorable, professional, and meaningful.

### Knowledge Graph

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul, essence | CRUD knowledge entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Triggers spreading activation on neighbors. |
| `cuba_cronica` | **Crónica** — chronicle | Attach observations to entities with **contradiction detection**, **dedup gate**, and **information density gating (V5)**. Supports `batch_add` with density parity. Types: `fact`, `decision`, `lesson`, `preference`, `context`, `tool_usage`. |
| `cuba_puente` | **Puente** — bridge | Connect entities with typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). **Traverse** walks the graph. **Infer** discovers transitive connections (A→B→C). |

### Search & Verification

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | Search with **4-signal RRF fusion** + **KG-neighbor expansion (V9)** when recall is low. **verify** mode with **source triangulation (V3)** and **adaptive confidence (V4)**. All results get **batch access tracking (V14)** for FSRS accuracy. **Token-budget truncation (V1)** + **post-fusion dedup (V2)**. Session-aware: boosts results matching active goals **(V13)**. |

### Error Memory

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_alarma` | **Alarma** — alarm | Report errors immediately. Auto-detects patterns (≥3 similar = warning). Hebbian boosting for retrieval. |
| `cuba_remedio` | **Remedio** — remedy | Mark an error as resolved. Cross-references similar unresolved errors. |
| `cuba_expediente` | **Expediente** — case file | Search past errors/solutions. **Anti-repetition guard**: warns if a similar approach previously failed. |

### Sessions & Decisions

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_jornada` | **Jornada** — workday | Track working sessions with goals and outcomes. Goals used for **session-aware decay (V6)** and **search boost (V13)**. |
| `cuba_decreto` | **Decreto** — decree | Record architecture decisions with context, alternatives, and rationale. |

### Memory Maintenance

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_zafra` | **Zafra** — sugar harvest 🎯 | Memory consolidation: **decay** (FSRS v4 adaptive, session-aware V6), **prune**, **merge**, **summarize**, **pagerank** (personalized), **find_duplicates**, **export** (bounded JSON backup), **stats**, **backfill**. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: **positive** (Oja's rule boost), **negative** (decrease), **correct** (update with versioning). |
| `cuba_vigia` | **Vigía** — watchman | Graph analytics: **summary** (counts + token estimate), **health** (staleness, Shannon entropy, MTTR, DB size), **drift** (chi-squared), **communities** (Louvain + **V12 summaries** for ≥3 members), **bridges** (betweenness centrality). |

---

## Mathematical Foundations

Cuba-Memorys is built on peer-reviewed algorithms, not ad-hoc heuristics:

### FSRS-6 Adaptive Decay — Ye (2024) / Expertium (2025)

```
R(t, S) = (1 + FACTOR · t/S)^(-decay_rate)
S_new = S · (1 + e^0.1 · (11 - D) · S^(-0.2) · (e^((1-R)·0.9) - 1))
```

**v2.0 upgrade**: FSRS-6 with 21 parameters (up from 4 in v4). Per-entity adaptive decay rate `w20` via sigmoid of `access_count` — frequently accessed memories decay slower. Active session goals exempt observations from decay.

### Oja's Rule (1982) — Hebbian Learning

```
Positive: Δw = η · (1 - w²)     → converges to 1.0, cannot explode
Negative: Δw = η · (1 + w²)     → converges to 0.01 (floor)
```

Where `η = 0.05`. The `w²` term provides natural saturation — self-normalizing without explicit clipping.

### TF-IDF + RRF Fusion — Salton (1975) / Cormack (2009)

```
tfidf(t, d) = tf(t, d) · log(N / df(t))
RRF(d) = Σ 1/(k + rank_i(d))     where k = 60
```

Reciprocal Rank Fusion combines multiple ranked lists from independent signals into a single robust ranking. **V2**: Post-fusion deduplication removes semantic duplicates via word-overlap ratio.

### KG-Neighbor Query Expansion (V9) — Zep/Graphiti (2025)

```
R₁ = φ(q)                                    # Initial search
q' = q ∪ {e.name : e ∈ neighbors(R₁, depth=1)}  # Expand via KG
R₂ = φ_trgm(q')                              # Re-search
Final = RRF(R₁, R₂)                          # Fuse
```

Inspired by Zep/Graphiti BFS search (arXiv 2501.13956 §3.1). Only triggered when initial results < limit (low recall). Caps neighbor names to 5 to prevent query bloat. Literature shows +15-25% recall improvement.

### Source Triangulation (V3) — Multi-Source Confidence

```
diversity = unique_entities / total_evidence
confidence_adjusted = confidence × (0.7 + 0.3 × diversity)
```

Penalizes single-source evidence to reduce confirmation bias. Multiple independent entities providing evidence increases confidence.

### Shannon Entropy — Information Density Gating (V5)

```
H = -Σ (c/n) · log₂(c/n)
density = H / H_max     ∈ [0, 1]
```

Observations with `density < 0.3` get reduced initial importance (0.3 instead of 0.5). Prevents low-information noise from polluting the knowledge base. Applied in both `add` and `batch_add` with parity.

### Optional BGE Embeddings — BAAI (2023)

```
model: Qdrant/bge-small-en-v1.5-onnx-Q (quantized, ~130MB)
runtime: ONNX (no PyTorch dependency)
similarity: cosine(embed(query), embed(observation))
```

Auto-downloads on first use. Falls back to TF-IDF if not installed. **V10**: LRU cache (256 entries) eliminates redundant ONNX calls (~84% reduction per session).

### GraphRAG Enrichment

Top-3 search results are enriched with degree-1 graph neighbors via a single batched SQL query. Each result gets a `graph_context` array containing neighbor name, entity type, relation type, and Hebbian strength.

### REM Sleep Daemon

After 15 minutes of user inactivity, an autonomous consolidation coroutine runs:

1. **FSRS Decay** — Applies memory decay using Ye (2023) v4 algorithm. **V6**: Excludes observations linked to active session goals
2. **Prune** — Removes low-importance (< 0.1), rarely-accessed observations
3. **PageRank** — Recalculates personalized importance scores
4. **Relation Decay** — Weakens unused relation strengths
5. **TF-IDF Rebuild** — Refreshes the TF-IDF index
6. **Cache Clear** — Clears search cache + **V10 embedding LRU cache**

Cancels immediately on new user activity. Prevents concurrent runs.

### Conditional pgvector

```
IF pgvector extension detected:
  → Migrate embedding column: float4[] → vector(384)
  → Create HNSW index (m=16, ef_construction=64, vector_cosine_ops)
  → Add vector cosine distance as 4th RRF signal
  → Persist embeddings on observation insert (async, V7/V11)
ELSE:
  → Graceful degradation: TF-IDF + trigrams (unchanged)
```

Zero-downtime: auto-detects at startup, no configuration needed.

### Personalized PageRank — Brin & Page (1998)

```
PR(v) = (1-α)/N + α · Σ PR(u)/deg(u)     where α = 0.85
personalization: biased toward recently active entities
final_importance = 0.6·PR + 0.4·current_importance
```

### Community Detection + Summaries (V12)

```
communities = Louvain(G, resolution=1.0)     — Blondel et al. (2008)
```

For communities ≥3 entities, **V12** generates structured summaries from top-3 observations per entity (by importance). Inspired by Zep/Graphiti hierarchical KG (arXiv 2501.13956 §2.3).

### Other Algorithms

- **Shannon Entropy** — Knowledge diversity scoring (`diversity = H/H_max ∈ [0,1]`)
- **Spreading Activation** — Collins & Loftus (1975): neighbors get 0.6% importance boost per access
- **Chi-Squared Drift** — Pearson (1900): detects error type distribution changes
- **Contradiction Detection** — TF-IDF overlap (>0.7) + negation patterns (EN + ES)

---

## Architecture

v2.0 is a **complete Rust rewrite** with modular architecture:

```
cuba-memorys/
├── docker-compose.yml          # Dedicated PostgreSQL (port 5488)
├── README.md
├── rust/                       # ← PRIMARY IMPLEMENTATION (v2.0)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs             # mimalloc + graceful shutdown
│   │   ├── protocol.rs         # JSON-RPC 2.0 + REM daemon (4h cycle)
│   │   ├── handlers/           # 12 MCP tool handlers (12 files)
│   │   ├── cognitive/          # FSRS-6, Hebbian/BCM, Dual-Strength, density
│   │   ├── search/             # RRF, confidence, LRU cache
│   │   ├── graph/              # Brandes centrality, Leiden, PageRank
│   │   └── embeddings/         # ONNX BGE-small (optional)
│   └── tests/
└── src/cuba_memorys/            # Python legacy (v1.6.0)
```

### Performance: Rust vs Python

| Metric | Python v1.6.0 | Rust v2.0 |
| ------ | :-----------: | :-------: |
| Binary size | ~50MB (venv) | **7.6MB** |
| Entity create | ~2ms | **498µs** |
| Hybrid search | <5ms | **2.52ms** |
| Analytics | <2.5ms | **958µs** |
| Memory usage | ~120MB | **~15MB** |
| Startup time | ~2s | **<100ms** |
| Dependencies | 12 Python packages | **0 runtime deps** |

### Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | Knowledge graph nodes | tsvector + pg_trgm indexes, importance ∈ [0,1], FSRS stability/difficulty |
| `brain_observations` | Facts attached to entities | 9 types, provenance, versioning, temporal validity, `vector(384)` embedding (if pgvector) |
| `brain_relations` | Graph edges | 5 types, bidirectional delete, Hebbian strength |
| `brain_errors` | Error memory | JSONB context, synapse weight, MTTR tracking |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking |

### Search Pipeline

Cuba-Memorys uses **Reciprocal Rank Fusion (RRF, k=60)** to combine up to 4 independent ranked signals:

| # | Signal | Source | Condition |
|---|--------|--------|-----------|
| 1 | Entities (ts_rank + trigrams + importance + freshness) | `brain_entities` | Always |
| 2 | Observations (ts_rank + trigrams + TF-IDF + importance) | `brain_observations` | Always |
| 3 | Errors (ts_rank + trigrams + synapse_weight) | `brain_errors` | Always |
| 4 | **Vector cosine distance (HNSW)** | `brain_observations.embedding` | pgvector installed |

Each signal produces an independent ranking. RRF fuses them: `score(d) = Σ 1/(60 + rank_i(d))`.

**Post-fusion pipeline:**
1. **V2 Deduplication** — Removes semantic duplicates via word-overlap ratio
2. **V9 KG-neighbor expansion** — When results < limit, expands query via graph neighbors
3. **V13 Session boost** — Active session goal keywords boost matching results by 15%
4. **GraphRAG enrichment** — Top-3 results get degree-1 neighbor context
5. **V1 Token-budget truncation** — Character budget allocated proportionally to RRF score
6. **V14 Batch access tracking** — All returned observations get `access_count` bump for FSRS accuracy

### Dependencies

**Rust v2.0:**
- `tokio` — Async runtime
- `sqlx` — PostgreSQL (async, compile-time checked)
- `pgvector` — Vector similarity search
- `ort` — ONNX Runtime (optional)
- `blake3` — Cryptographic hashing
- `mimalloc` — Global allocator
- `tracing` — Structured logging

**Python v1.6.0 (legacy):**
- `asyncpg`, `orjson`, `scikit-learn`, `networkx`, `scipy`, `rapidfuzz`, `numpy`
- Optional: `onnxruntime`, `huggingface-hub`, `tokenizers`

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |

### Docker Compose

Runs a dedicated PostgreSQL 18 Alpine instance:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always (auto-starts on boot)
- **Healthcheck**: `pg_isready` every 10s

---

## How It Works in Practice

### 1. The agent learns from your project

```
Agent: I learned that FastAPI endpoints must use async def with response_model.
→ cuba_alma(create, "FastAPI", technology)
→ cuba_cronica(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: I got IntegrityError: duplicate key on numero_parte.
→ cuba_alarma("IntegrityError", "duplicate key on numero_parte")
→ Similar error found! Solution: "Add SELECT EXISTS before INSERT with FOR UPDATE"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify this claim before responding...
→ cuba_faro("FastAPI uses Django ORM", mode="verify")
→ confidence: 0.0, level: "unknown"
→ recommendation: "No supporting evidence found. High hallucination risk."
```

### 4. Memories adapt with FSRS v4

```
Initial stability:    S = 1.0 (decays in ~9 days)
After 5 reviews:      S = 8.2 (decays in ~74 days)
After 20 reviews:     S = 45.0 (survives ~13 months)
```

### 5. KG-neighbor query expansion (V9)

```
Query: "deployment strategy"    → 2 results (below limit of 5)
→ KG neighbors: [Docker, CI/CD, NGINX]
→ Expanded: "deployment strategy Docker CI/CD NGINX"
→ 5 results (recall improved by graph context)
```

### 6. Community summaries (V12)

```
→ cuba_vigia(metric="communities")
→ Community 0 (4 members): [FastAPI, Pydantic, SQLAlchemy, PostgreSQL]
  Summary: "FastAPI: All endpoints async def; Pydantic: V2 strict models; ..."
→ Community 1 (3 members): [React, Next.js, TypeScript]
  Summary: "React: 19 hooks patterns; Next.js: App Router; ..."
```

---

## Verification

Tested with NEMESIS protocol (3-tier) — v1.6.0 results:

```
🟢 Normal (12/12)  — All 12 tools, all CRUD actions, GraphRAG enrichment,
                     Hebbian cross-reference, RRF fusion, Oja's rule,
                     dedup gate, anti-repetition guard, batch_add,
                     V14 access_count bump confirmed, V12 community summaries,
                     V9 KG expansion, V10 embedding cache transparent
🟡 Pessimist (4/4) — Empty queries, non-existent entities, duplicate observations
                     (dedup gate: similarity 0.85), ungrounded claims
                     (confidence: 0.0, level: "unknown", hallucination warning)
🔴 Extreme (3/3)   — SQL injection (parametrized queries — stored as text, not executed),
                     XSS (HTML entities escaped), repetitive content AAA×430
                     (density gating: importance reduced to 0.3)
```

Previous versions: v1.3.0 (12/12, 8/8, 9/9), v1.1.0 (13/13), v1.0.1 (18/18).

### Performance

| Operation | Avg latency |
| --------- | :---------: |
| RRF hybrid search | < 5ms |
| Analytics | < 2.5ms |
| Entity CRUD | < 1ms |
| PageRank (100 entities) | < 50ms |
| GraphRAG enrichment | < 2ms |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **2.0.0** | 🦀 **Complete Rust rewrite**. FSRS-6 (21 params), BCM metaplasticity, Dual-Strength memory, Brandes centrality, Leiden communities, Shannon entropy analytics, GraphRAG enrichment, token truncation, overload warnings, 3-range entropy RRF, adaptive k, source diversity scoring, sub-ms handlers |
| **1.6.0** | V9 KG-neighbor query expansion, V10 embedding LRU cache, V11 async rebuild_embeddings, V12 community summaries, V14 batch access tracking, V15 pool race fix |
| **1.5.0** | V1 token-budget truncation, V2 post-fusion dedup, V3 source triangulation, V4 adaptive confidence, V5 batch_add density parity, V6 session-aware decay, V7 async embed, V8 temporal index |
| **1.3.0** | Modular architecture (CC avg D→A), 87% CC reduction |
| **1.1.0** | GraphRAG, REM Sleep, conditional pgvector, 4-signal RRF |
| **1.0.0** | Initial release: 12 tools, Hebbian learning, FSRS |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Wozniak (1987), Ye (2023, FSRS v4), Oja (1982), Salton (1975, TF-IDF), Cormack (2009, RRF), Brin & Page (1998, PageRank), Collins & Loftus (1975), Shannon (1948), Pearson (1900, χ²), Blondel et al. (2008, Louvain), McCabe (1976, CC), BAAI (2023, BGE), Malkov & Yashunin (2018, HNSW), Karpicke & Roediger (2008, testing effect), Zep/Graphiti (2025, arXiv 2501.13956).
