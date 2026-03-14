# 🧠 Cuba-Memorys

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/LeandroPG19/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Audit](https://img.shields.io/badge/audit-GO-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, neuroscience-inspired algorithms, and anti-hallucination grounding.

12 tools with Cuban soul. Sub-millisecond handlers. Mathematically rigorous.

> [!IMPORTANT]
> **v2.0.0** — Complete Rust rewrite. 5,452 LOC, 36 source files, 0 CVEs, 0 tech debt. Internally audited with GO verdict. The Python version (v1.6.0) remains in `src/` as legacy reference.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this:

- 🧬 **FSRS-6 adaptive decay** — Memories strengthen with use and fade realistically (21 parameters, per-entity sigmoid)
- 🧠 **Hebbian + BCM metaplasticity** — Self-normalizing importance via Oja's rule with Bienenstock-Cooper-Munro throttle
- 🔍 **4-signal RRF fusion search** — TF-IDF + pg_trgm + full-text + pgvector HNSW, with entropy-routed adaptive k
- 🕸️ **Knowledge graph** — Entities, observations, typed relations with Leiden community detection
- 🛡️ **Anti-hallucination grounding** — Verify claims against stored knowledge with graduated confidence scoring
- 💤 **REM Sleep consolidation** — Autonomous background decay + PageRank + neighbor diffusion after idle
- 📊 **Graph intelligence** — Personalized PageRank, Leiden communities, Brandes centrality, Shannon entropy
- 🔐 **Dual-Strength memory** — Storage + retrieval strengths track independently (Bjork 1992)
- ⚡ **Error memory** — Never repeat the same mistake (anti-repetition guard)

### Comparison

| Feature | Cuba-Memorys | Basic Memory MCPs |
| ------- | :----------: | :---------------: |
| Knowledge graph with typed relations | ✅ | ❌ |
| FSRS-6 adaptive spaced repetition | ✅ | ❌ |
| Hebbian learning + BCM metaplasticity | ✅ | ❌ |
| Dual-Strength memory model | ✅ | ❌ |
| 4-signal entropy-routed RRF fusion | ✅ | ❌ |
| KG-neighbor query expansion | ✅ | ❌ |
| GraphRAG topological enrichment | ✅ | ❌ |
| Leiden community detection | ✅ | ❌ |
| Brandes betweenness centrality | ✅ | ❌ |
| Shannon entropy analytics | ✅ | ❌ |
| Adaptive prediction error gating | ✅ | ❌ |
| Anti-hallucination verification | ✅ | ❌ |
| Error pattern detection + MTTR | ✅ | ❌ |
| Session-aware search boost | ✅ | ❌ |
| REM Sleep autonomous consolidation | ✅ | ❌ |
| Embedding LRU cache | ✅ | ❌ |
| Optional ONNX BGE embeddings | ✅ | ❌ |
| Write-time dedup gate | ✅ | ❌ |
| Contradiction auto-supersede | ✅ | ❌ |
| Graceful shutdown (SIGTERM/SIGINT) | ✅ | ❌ |

---

## Quick Start

```bash
git clone https://github.com/LeandroPG19/cuba-memorys.git
cd cuba-memorys

# Start PostgreSQL
docker compose up -d

# Build Rust binary
cd rust
cargo build --release
```

Configure your AI editor (Claude Code, Cursor, Windsurf, etc.):

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

### Optional: ONNX Embeddings

For real BGE-small-en-v1.5 semantic embeddings instead of hash-based fallback:

```bash
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding.

---

## 🇨🇺 The 12 Tools

Every tool is named after Cuban culture — memorable, professional, meaningful.

### Knowledge Graph

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul | CRUD entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Triggers Hebbian + Dual-Strength boost. |
| `cuba_cronica` | **Crónica** — chronicle | Attach observations with **dedup gate**, **contradiction detection**, **Shannon density gating**, and **adaptive PE gating V5.1**. Supports `batch_add`. |
| `cuba_puente` | **Puente** — bridge | Typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). **Traverse** walks the graph. **Infer** discovers transitive paths. blake3 dedup. |

### Search & Verification

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | 4-signal RRF fusion with entropy routing and adaptive k. KG-neighbor expansion for low recall. `verify` mode with source triangulation. Session-aware boost. |

### Error Memory

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alarma` | **Alarma** — alarm | Report errors. Auto-detects patterns (≥3 similar = warning). |
| `cuba_remedio` | **Remedio** — remedy | Resolve errors with cross-reference to similar unresolved issues. |
| `cuba_expediente` | **Expediente** — case file | Search past errors. **Anti-repetition guard**: warns if similar approach failed before. |

### Sessions & Decisions

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_jornada` | **Jornada** — workday | Session tracking with goals and outcomes. Goals used for decay exemption and search boost. |
| `cuba_decreto` | **Decreto** — decree | Record architecture decisions with context, alternatives, rationale. |

### Memory Maintenance

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_zafra` | **Zafra** — sugar harvest | Consolidation: FSRS-6 decay, prune, merge, summarize, pagerank, find_duplicates, export, stats, backfill. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: positive (Oja boost), negative (decrease), correct (update with versioning). |
| `cuba_vigia` | **Vigía** — watchman | Analytics: summary, health (Shannon entropy, MTTR), drift (χ²), Leiden communities, Brandes bridges. |

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml           # Dedicated PostgreSQL 18 (port 5488)
├── rust/                        # ← PRIMARY (v2.0.0, 5452 LOC)
│   ├── src/
│   │   ├── main.rs              # mimalloc + graceful shutdown
│   │   ├── protocol.rs          # JSON-RPC 2.0 + REM daemon (4h cycle)
│   │   ├── db.rs                # sqlx PgPool (10 max, 600s idle, 1800s lifetime)
│   │   ├── schema.sql           # 5 tables, 15+ indexes, HNSW
│   │   ├── constants.rs         # Tool definitions, thresholds, enums
│   │   ├── handlers/            # 12 MCP tool handlers (1 file each)
│   │   ├── cognitive/           # FSRS-6, Hebbian/BCM, Dual-Strength, PE gating
│   │   ├── search/              # RRF fusion, BM25+, confidence, LRU cache
│   │   ├── graph/               # Brandes centrality, Leiden, PageRank
│   │   └── embeddings/          # ONNX BGE-small (optional, spawn_blocking)
│   └── tests/
└── src/cuba_memorys/            # Python legacy (v1.6.0)
```

### Performance: Rust vs Python

| Metric | Python v1.6.0 | Rust v2.0.0 |
| ------ | :-----------: | :---------: |
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
| `brain_entities` | KG nodes | tsvector + pg_trgm + GIN indexes, importance, FSRS stability/difficulty |
| `brain_observations` | Facts with provenance | 9 types, versioning, temporal validity, Dual-Strength, `vector(384)` (pgvector) |
| `brain_relations` | Typed edges | 5 types, bidirectional, Hebbian strength, blake3 dedup |
| `brain_errors` | Error memory | JSONB context, synapse weight, MTTR, pattern detection |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking, duration |

### Search Pipeline

4-signal **Reciprocal Rank Fusion (RRF, k=60)** with entropy-routed weighting:

| # | Signal | Source | Condition |
|---|--------|--------|-----------| 
| 1 | Entities (ts_rank + trigrams + importance + freshness) | `brain_entities` | Always |
| 2 | Observations (ts_rank + trigrams + BM25+ + importance) | `brain_observations` | Always |
| 3 | Errors (ts_rank + trigrams + synapse_weight) | `brain_errors` | Always |
| 4 | **Vector cosine distance (HNSW)** | `brain_observations.embedding` | pgvector installed |

**Post-fusion pipeline:** Dedup → KG-neighbor expansion → Session boost → GraphRAG enrichment → Token-budget truncation → Batch access tracking

---

## Mathematical Foundations

Built on peer-reviewed algorithms, not ad-hoc heuristics:

### FSRS-6 — Ye (2024) / Expertium (2025)
```
R(t, S) = (1 + FACTOR · t/S)^(-decay_rate)
S_new = S · (1 + e^0.1 · (11 - D) · S^(-0.2) · (e^((1-R)·0.9) - 1))
```
21 parameters. Per-entity adaptive decay via sigmoid of `access_count` — frequently accessed memories decay slower.

### Hebbian + BCM — Oja (1982), Bienenstock-Cooper-Munro (1982)
```
Positive: Δw = η · (1 - w²)    → self-normalizing, converges to 1.0
Negative: Δw = η · (1 + w²)    → floor at 0.01
BCM: θ_M slides based on recent activity → prevents winner-take-all
```

### Dual-Strength — Bjork (1992)
```
Storage:   S_s = S_s + α·(1 - S_s)     (grows fast, never decreases)
Retrieval: S_r = S_r + β·(1 - S_r)·S_s  (grows proportional to storage)
Effective: importance = 0.4·S_s + 0.6·S_r
```

### RRF Fusion — Cormack (2009)
```
RRF(d) = Σ w_i/(k + rank_i(d))   where k = 60
```
Entropy-routed weighting: keyword-dominant vs mixed vs semantic queries get different signal weights.

### Other Algorithms

| Algorithm | Reference | Used in |
|-----------|-----------|---------|
| **BM25+** (δ=1) | Lv & Zhai (SIGIR 2011) | `tfidf.rs` |
| **Leiden communities** | Traag et al. (Nature 2019) | `community.rs` → `vigia.rs` |
| **Personalized PageRank** | Brin & Page (1998) | `pagerank.rs` → `zafra.rs` |
| **Brandes centrality** | Brandes (2001) | `centrality.rs` → `vigia.rs` |
| **Neighbor diffusion** | Collins & Loftus (1975) | `spreading.rs` → REM daemon |
| **Adaptive PE gating** | Friston (Nature 2023) | `prediction_error.rs` → `cronica.rs` |
| **Shannon entropy** | Shannon (1948) | `density.rs` → information gating |
| **Chi-squared drift** | Pearson (1900) | Error distribution change detection |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (**required**) |
| `ONNX_MODEL_PATH` | — | Path to BGE model directory (optional) |
| `ORT_DYLIB_PATH` | — | Path to libonnxruntime.so (optional) |
| `RUST_LOG` | `cuba_memorys=info` | Log level filter |

### Docker Compose

Dedicated PostgreSQL 18 Alpine:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always
- **Healthcheck**: `pg_isready` every 10s

---

## How It Works

### 1. The agent learns from your project

```
Agent: FastAPI requires async def with response_model.
→ cuba_alma(create, "FastAPI", technology)
→ cuba_cronica(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: IntegrityError: duplicate key on numero_parte.
→ cuba_alarma("IntegrityError", "duplicate key on numero_parte")
→ cuba_expediente: Similar error found! Solution: "Add SELECT EXISTS before INSERT"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify before responding...
→ cuba_faro("FastAPI uses Django ORM", mode="verify")
→ confidence: 0.0, level: "unknown" — "No evidence. High hallucination risk."
```

### 4. Memories adapt with FSRS-6

```
Initial stability:    S = 1.0  (decays in ~9 days)
After 5 reviews:      S = 8.2  (decays in ~74 days)
After 20 reviews:     S = 45.0 (survives ~13 months)
```

### 5. Community intelligence

```
→ cuba_vigia(metric="communities")
→ Community 0 (4 members): [FastAPI, Pydantic, SQLAlchemy, PostgreSQL]
  Summary: "Backend stack: async endpoints, V2 validation, 2.0 ORM..."
→ Community 1 (3 members): [React, Next.js, TypeScript]
  Summary: "Frontend stack: React 19, App Router, strict types..."
```

---

## Security & Audit

**Internal Audit Verdict: ✅ GO** (2026-03-14)

| Check | Result |
|-------|:------:|
| SQL injection | ✅ All queries parameterized (sqlx bind) |
| CVEs in dependencies | ✅ 0 active (sqlx 0.8.6, tokio 1.50.0) |
| UTF-8 safety | ✅ `safe_truncate` on all string slicing |
| Secrets | ✅ All via environment variables |
| Division by zero | ✅ Protected with `.max(1e-9)` |
| Error handling | ✅ All `?` propagated with `anyhow::Context` |
| Clippy pedantic | ✅ 276 cosmetic warnings only |
| Tests | ✅ 11/11 passing |
| Licenses | ✅ All MIT/Apache-2.0 (0 GPL/AGPL) |

---

## Dependencies

| Crate | Purpose | License |
|-------|---------|---------|
| `tokio` | Async runtime | MIT |
| `sqlx` | PostgreSQL (async) | MIT/Apache-2.0 |
| `serde` / `serde_json` | Serialization | MIT/Apache-2.0 |
| `pgvector` | Vector similarity | MIT |
| `petgraph` | Graph algorithms | MIT/Apache-2.0 |
| `ort` | ONNX Runtime (optional) | MIT/Apache-2.0 |
| `tokenizers` | HuggingFace tokenizers | Apache-2.0 |
| `blake3` | Cryptographic hashing | Apache-2.0/CC0 |
| `mimalloc` | Global allocator | MIT |
| `tracing` | Structured JSON logging | MIT |
| `lru` | O(1) LRU cache | MIT |
| `chrono` | Timezone-aware timestamps | MIT/Apache-2.0 |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **2.0.0** | 🦀 Complete Rust rewrite. FSRS-6 (21 params), BCM metaplasticity, Dual-Strength memory, Brandes centrality, Leiden communities, Shannon entropy, BM25+, adaptive RRF, neighbor diffusion, blake3 dedup. Internal audit: GO verdict. |
| **1.6.0** | KG-neighbor expansion, embedding LRU cache, async embed rebuild, community summaries, batch access tracking |
| **1.5.0** | Token-budget truncation, post-fusion dedup, source triangulation, adaptive confidence, session-aware decay |
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

Mathematical foundations: Wozniak (1987), Ye (2024, FSRS-6), Expertium (2025), Oja (1982), Bienenstock, Cooper & Munro (1982, BCM), Bjork (1992, Dual-Strength), Salton (1975, TF-IDF), Lv & Zhai (2011, BM25+), Cormack (2009, RRF), Brin & Page (1998, PageRank), Collins & Loftus (1975), Traag et al. (2019, Leiden), Brandes (2001), Shannon (1948), Pearson (1900, χ²), Friston (2023, PE gating), BAAI (2023, BGE), Malkov & Yashunin (2018, HNSW), O'Connor et al. (2020, blake3).
