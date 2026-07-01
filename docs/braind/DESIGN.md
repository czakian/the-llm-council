# braind — Design Document

**Status:** Draft v1 (design only; implementation lives in a future `czakian/braind` repository)
**Date:** 2026-07-01

`braind` is a per-host daemon, written in Rust, that manages a set of embedded knowledge bases
("**brains**") for coding agents. Agents interact with brains over **MCP (streamable HTTP)** using
four core verbs — **learn, think, relearn, forget** — plus attachment, export/import, and
maintenance tools. Storage is **LanceDB**, retrieval is two-stage (**dense ANN → ColBERT
late-interaction reranking**), embeddings run in-process via **HuggingFace models on Candle**, and
every mutation is journaled to auditable, portable files.

---

## 1. Goals and requirements traceability

| # | Requirement (as stated) | Where addressed |
|---|--------------------------|-----------------|
| 1 | CRUD via `/learn`, `/think`, `/relearn`, `/forget` skills | §10 Skills, §9 MCP tool surface |
| 2 | Multiple agents interact with brains over MCP | §7 MCP server, §8 Sessions |
| 3 | Attach to multiple brains via auto-discovery + explicit command; attachment propagates to sub-session agents | §8 Attachment & discovery |
| 4 | Read-after-write per brain, multi-reader/single-writer, eventual consistency across brains | §6 Consistency model |
| 5 | Export/import all knowledge as auditable JSON/Markdown, portable across hosts | §11 Journal, export/import |
| 6 | ColBERT reranking, 700+-dim embeddings | §5 Retrieval pipeline |
| 7 | Rust MCP server over HTTP, HuggingFace transformers | §7 Server, §5.2 Embedding runtime |
| 8 | One server per host; all agents attach to it over HTTP | §7.1 Topology |
| 9 | Full lifecycle: prune, deepen, re-ground, dedupe, synthesize/compact via cadence-spawned background agents | §12 Lifecycle management |

### Non-goals (v1)

- Multi-writer distributed brains (one host = one writer; cross-host is journal replication, §11.4).
- Serving non-local clients by default (localhost-only unless explicitly configured, §13.2).
- A general RAG product. This is purpose-built agent memory for coding workflows.

---

## 2. System overview

```
┌────────────────────────────── host ──────────────────────────────┐
│                                                                   │
│  Claude Code session A ──┐                                        │
│    ├─ subagent A1 ───────┤   MCP over streamable HTTP             │
│    └─ subagent A2 ───────┤   http://127.0.0.1:27246/mcp           │
│  Claude Code session B ──┤   (X-Braind-Session header)            │
│  background maint agents ┘                                        │
│              │                                                    │
│              ▼                                                    │
│  ┌───────────────────────── braind ──────────────────────────┐    │
│  │  axum + rmcp (streamable HTTP server)                     │    │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────────┐              │    │
│  │  │ session  │ │  brain    │ │  scheduler   │              │    │
│  │  │ registry │ │  registry │ │  (lifecycle) │              │    │
│  │  └──────────┘ └─────┬─────┘ └──────┬───────┘              │    │
│  │        ┌────────────┴─────┐        │ spawns `claude -p` / │    │
│  │        ▼                  ▼        ▼ agent CLI jobs       │    │
│  │  ┌───────────┐   ┌──────────────────┐                     │    │
│  │  │ embed     │   │ per-brain store   │                    │    │
│  │  │ service   │   │ Lance dataset +   │                    │    │
│  │  │ (candle)  │   │ RwLock + journal  │                    │    │
│  │  └───────────┘   └──────────────────┘                     │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ~/.braind/                                                       │
│    config.toml, token                                             │
│    models/            (hf-hub cache: encoder + ColBERT)           │
│    brains/<name>/     data.lance/  journal/  exports/  brain.toml │
└───────────────────────────────────────────────────────────────────┘
```

One `braind` process per host owns all brain data on that host. Everything else — interactive
sessions, spawned subagents, background maintenance agents — is an MCP client of that one process.
That single-owner rule is what makes the consistency story (§6) simple.

---

## 3. Core concepts

### 3.1 Brain

A named knowledge base with its own Lance dataset, journal, config, and lock. Brains come in three
scopes:

| Scope | Examples | Naming | Lives at |
|-------|----------|--------|----------|
| **Global** | `perf`, `build-test`, `prod-metrics` | free-form slug | `~/.braind/brains/<slug>/` |
| **Project** | one per repo/worktree | `project--<slug>-<hash8(repo-root)>` | `~/.braind/brains/…` (data) + optional in-repo `.braind/` (config + committed exports) |
| **Plan** | one per plan/initiative | `plan--<plan-id>` | `~/.braind/brains/…`, parented to a project brain |

Plan brains carry a `parent` pointer to their project brain; project brains may declare default
global brains to co-attach (e.g. a Rust project auto-attaches `perf` and `build-test`).

### 3.2 Record

The unit of knowledge. Stored as one Lance row + one journal entry per mutation.

```jsonc
{
  "id": "01J9Z6K8Q2W3E4R5T6Y7U8I9O0",     // ULID — sortable, host-unique
  "brain": "perf",
  "kind": "insight",                       // fact | procedure | insight | decision | metric | episode
  "title": "tokio mpsc beats crossbeam for our fan-in shape",
  "body": "…markdown…",                    // the knowledge itself
  "tags": ["tokio", "channels", "benchmarks"],
  "evidence": [                            // what grounds this record (used by re-ground jobs, §12)
    {"type": "file",   "ref": "bench/fanin.rs@9f3c2e1", "note": "benchmark source"},
    {"type": "command","ref": "cargo bench --bench fanin", "expect": "mpsc ≥ 1.4x"},
    {"type": "url",    "ref": "https://…"}
  ],
  "confidence": 0.9,                       // 0..1, decays and is restored by verification
  "created_at": "2026-07-01T12:00:00Z",
  "updated_at": "2026-07-01T12:00:00Z",
  "last_verified_at": "2026-07-01T12:00:00Z",
  "access_count": 3,                       // bumped by think() hits; drives prune/deepen
  "supersedes": null,                      // relearn chains: new record points at old
  "superseded_by": null,
  "tombstone": false,                      // forget() sets true; purge removes the row (journal keeps it)
  "origin": {"host": "devbox-1", "agent": "claude-code", "session": "s_…"}
}
```

Plus two vector columns (not in the JSON export of the *record*, recomputable):

- `vector: FixedSizeList<f32, D>` — dense document embedding (D ≥ 768, §5.1)
- `token_vectors: List<FixedSizeList<f32, Dt>>` — ColBERT per-token embeddings for MaxSim reranking

Long bodies are chunked (~512-token chunks with overlap) into child rows sharing a `record_id`;
`think()` retrieves chunks but returns whole records.

### 3.3 Session

A server-side context created by `attach`. It maps a session id → the set of attached brains + a
default *write target*. All MCP calls carry the session id; subagents inherit it (§8.3).

---

## 4. Storage layer — LanceDB

**Choice: Lance/LanceDB via the `lancedb` Rust crate**, embedded (no separate DB server).

Why Lance over the alternatives:

| | LanceDB (embedded) | Qdrant/Milvus (server) | sqlite-vec / pgvector |
|---|---|---|---|
| Rust-native, in-process | ✅ crate | ❌ another daemon | ✅ / ❌ |
| MVCC + dataset versioning (time travel) | ✅ built-in | partial | ❌ |
| ANN indexes (IVF_PQ, HNSW) | ✅ | ✅ | limited |
| Multivector / late-interaction storage | ✅ (list-of-vector columns; native MaxSim in newer versions) | ✅ Qdrant only | ❌ |
| Ops burden for "one daemon per host" | none | high | low |

Layout: **one Lance dataset per brain** at `~/.braind/brains/<name>/data.lance`. Per-brain
datasets (rather than one big table with a `brain` column) give us:

- independent locks → MRSW per brain falls out naturally (§6),
- independent compaction/index rebuilds,
- trivially portable brains (a brain directory is self-contained),
- per-brain embedding-model pinning (a brain records which model/dim produced its vectors).

Lance's atomic manifest-swap commits give crash safety; its versioning gives us cheap "as-of"
reads for audit and safe concurrent reads during writes. Indexes: IVF_PQ (default) on `vector`
once a brain exceeds ~10k rows; brute-force below that (exact and fast enough). Scalar indexes on
`tags`, `kind`, `tombstone`, `updated_at` for filtered retrieval.

---

## 5. Retrieval pipeline

### 5.1 Models

Two models, both pulled from the HuggingFace Hub (via `hf-hub`) into `~/.braind/models/` on first
run, pinned by revision in config:

| Role | Default | Dim | Why |
|------|---------|-----|-----|
| Dense encoder | **`BAAI/bge-m3`** | **1024** (satisfies 700+) | strong retrieval quality, 8k context, multilingual, permissive license |
| ColBERT reranker | **`answerdotai/answerai-colbert-small-v1`** | 96/token | ~33M params — fast enough to encode queries per-call on CPU; near-SOTA rerank quality |
| (alt encoder) | `nomic-ai/nomic-embed-text-v1.5` | 768 (Matryoshka) | smaller/faster option, still ≥ 700 |
| (alt reranker) | `jinaai/jina-colbert-v2` | 128/token | multilingual, 8k context, heavier |

The encoder is a **per-brain** setting recorded in `brain.toml` and the export manifest — you can
never mix vectors from different models in one brain; import re-embeds when models differ (§11.3).

### 5.2 Embedding runtime

**Candle** (HuggingFace's Rust ML framework) as the primary runtime — pure Rust, loads safetensors
straight from the Hub, and supports the BERT/XLM-R families both models belong to. Feature-flagged
`cuda`/`metal` backends; CPU is the baseline and is sized to be adequate (embedding happens at
write time and at query time for one query string — not in bulk on the hot path).

**Fallback (feature flag `ort`):** `fastembed-rs`/ONNX Runtime for the same models, if a candle
gap appears for a chosen architecture. The `Embedder` trait isolates this choice:

```rust
#[async_trait]
trait Embedder: Send + Sync {
    fn id(&self) -> &ModelId;            // model + revision + dim, stamped into brain manifests
    fn dim(&self) -> usize;
    async fn embed_docs(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

#[async_trait]
trait LateInteractionEncoder: Send + Sync {
    async fn encode_doc(&self, text: &str) -> Result<TokenMatrix>;   // [tokens × Dt]
    async fn encode_query(&self, text: &str) -> Result<TokenMatrix>;
}
```

A single **embed service** task owns the models (loaded once, shared by all brains) and serves
requests over an internal mpsc queue with micro-batching (batch up to 16 docs / 10ms window) so
concurrent agents don't thrash the model.

### 5.3 Two-stage retrieval (`think`)

```
query ──► embed_query (bge-m3) ──► Lance ANN top-K (K=100, filtered: tombstone=false, brains∈session)
      └─► encode_query (ColBERT)──► MaxSim rerank of K candidates ──► top-N (N=8 default)
                                        │
                                        └─ score = Σ_q max_d (q·d)  over stored token_vectors
```

- Document token vectors are computed **at write time** and stored in the `token_vectors` column,
  so reranking is pure math (a batched matmul + rowmax + sum — implemented in candle, no model
  forward pass for documents at query time). Only the *query* is ColBERT-encoded per call.
- If Lance's native multivector/MaxSim search is available in the Rust crate at build time we use
  it; otherwise MaxSim runs in-process over the K candidates. Either way the interface and scores
  are identical — this is an implementation detail behind `Reranker`.
- Final score blends rerank score with record priors: `score' = maxsim_z * w1 + confidence * w2 +
  recency_decay * w3` (weights in config; defaults 0.8/0.1/0.1).
- Cross-brain queries fan out per attached brain in parallel, then merge candidates *before* the
  rerank stage so ColBERT ranks one unified pool (per-brain score normalization via z-scores).
- Every returned record gets `access_count += 1` and a journal `access` entry (async, off the read
  path — reads never take the write lock; see §6).

---

## 6. Consistency model

Guarantees, in order of strength:

1. **Per-brain read-after-write.** A `learn`/`relearn`/`forget` acknowledged to a client is
   visible to every subsequent `think` on that brain, from any session.
2. **Per-brain multi-reader / single-writer.** Any number of concurrent readers; at most one
   in-flight write per brain.
3. **Cross-brain eventual consistency.** No transaction spans brains. Derived cross-brain state
   (e.g. a synthesis that reads `perf` and writes a project brain) is eventually consistent.
4. **Cross-host eventual consistency.** Journal shipping (§11.4); last-write-wins per record id
   with supersede chains preserved.

Mechanics — all downstream of the single-owner rule (§2):

- `braind` is the **only process** that opens brain datasets for write. An advisory file lock
  (`~/.braind/braind.lock`) enforces one daemon per host; `brainctl` talks to the daemon rather
  than touching data directly.
- Each brain holds a `tokio::sync::RwLock<BrainHandle>`. Writers take the write lock, append the
  journal entry (fsync), commit the Lance write (atomic manifest swap), bump the cached latest
  version, then release. Journal-before-commit means a crash between the two is repaired on
  startup by replaying the journal tail against the dataset version — the journal is the WAL.
- Readers take the read lock only long enough to clone the current dataset version handle, then
  search lock-free against that immutable snapshot. Lance MVCC means an in-progress write never
  disturbs them. Because the writer bumps the cached version before releasing the write lock, any
  read that *starts* after a write is acknowledged sees it → read-after-write.
- Access-count bumps and other read-side telemetry go through the writer queue asynchronously;
  they are allowed to lag (they're statistics, not knowledge).

---

## 7. MCP server

### 7.1 Topology

- **One `braind` per host** (requirement 8), listening on `http://127.0.0.1:27246/mcp` by default
  (27246 = "BRAIN" on a phone keypad; configurable). All agents on the host — interactive
  sessions, their subagents, and scheduler-spawned maintenance agents — connect to this one
  endpoint.
- Transport: **MCP streamable HTTP** via the official Rust SDK (**`rmcp`**, feature
  `transport-streamable-http-server`) mounted in an **axum** app. Streamable HTTP (not stdio) is
  what makes "many clients, one server" work, and gives us server→client progress notifications
  for long operations (export, maintenance).
- The same axum app also serves non-MCP endpoints: `GET /healthz`, `GET /metrics` (Prometheus),
  and `GET /ui` (later: tiny read-only brain browser).

### 7.2 Auth

Localhost-only bind by default. Clients present a bearer token from `~/.braind/token`
(0600, generated on first start). This keeps other local users out on shared machines and is the
hook for remote/TLS deployment later. Maintenance agents spawned by the scheduler get a scoped
token (allowed brains + expiry) so a runaway background agent can't touch unrelated brains.

### 7.3 Client configuration (Claude Code and friends)

```jsonc
// .mcp.json (project) or ~/.claude/mcp.json (user)
{
  "mcpServers": {
    "brains": {
      "type": "http",
      "url": "http://127.0.0.1:27246/mcp",
      "headers": { "Authorization": "Bearer ${BRAIND_TOKEN}" }
    }
  }
}
```

Subagents spawned by a session reuse the same server config and inherit the session id (§8.3), so
requirement 3's "every sub-session agent" attachment comes for free.

---

## 8. Brain attachment and discovery

### 8.1 Explicit attach

The `attach_brains` tool (or `/brains attach perf,build-test` skill) attaches named brains to the
calling session. `create_brain` makes new ones. A session's attach set has one **write target**
(default: the project brain if present, else the first attached brain); `learn` takes an optional
`brain` argument to override per call.

### 8.2 Auto-discovery

`attach_brains { auto: true, cwd: "/path/to/repo" }` resolves, in order:

1. **Project brain** — walk up from `cwd` to the repo root; identity is
   `project--<slug>-<hash8(canonical repo root)>`. Created on first attach. If the repo contains
   `.braind/brain.toml`, its settings (name override, default co-attached brains, encoder choice)
   apply.
2. **Plan brains** — any brain with `parent == project brain` and `status == active`, or an
   explicit `plan` argument.
3. **Global defaults** — from `~/.braind/config.toml` `[defaults] attach = ["perf", "build-test"]`
   plus the project's `.braind/brain.toml` `attach = [...]` list.

Hosts stay authoritative for data; the *repo* only carries config + exports, so cloning a repo on
a new machine and attaching reconstructs the same brain topology (and `import` can seed it from
committed exports, §11).

### 8.3 Session inheritance

`attach_brains` returns a `session_id`. The client-side skill writes it to the environment
(`BRAIND_SESSION`) so every subagent the session spawns sends the same
`X-Braind-Session: <id>` header and lands in the same attach set — no per-subagent attach calls.
Sessions idle-expire after 24h (touching any tool renews); expiry only forgets the attach set,
never data.

### 8.4 Session-scoped tool behavior

All knowledge tools implicitly operate on the session's attached brains: `think` searches all of
them (with per-brain result attribution), `learn` writes to the write target, `forget`/`relearn`
address records by id (brain is derivable from the id via the registry).

---

## 9. MCP tool surface

| Tool | Args (essentials) | Behavior |
|------|-------------------|----------|
| `learn` | `title, body, kind?, tags?, evidence?, brain?, confidence?` | Embed (dense + ColBERT), journal, insert. Returns record id. Near-duplicate check first: if cosine ≥ 0.95 vs an existing record, returns `duplicate_of` instead of inserting (caller may force). |
| `think` | `query, n?, brains?, filter? (kind/tags/min_confidence), include_stale?` | Two-stage retrieval (§5.3) across attached brains. Returns records with scores, brain attribution, evidence, and staleness flags. |
| `relearn` | `id, patch {title?/body?/tags?/evidence?/confidence?} \| supersede_with {new record}` | In-place patch (re-embeds if body changed) or supersede: insert new record with `supersedes: id`, mark old `superseded_by`, journal both. |
| `forget` | `id \| filter, hard?` | Tombstone (default) — excluded from `think`, retained in Lance + journal for audit. `hard: true` deletes the row; the journal entry (with full prior record) remains the audit trail. |
| `attach_brains` | `brains? \| auto?, cwd?, plan?` | §8. Returns `session_id`, attach set, write target. |
| `detach_brains` | `brains` | Remove from session attach set. |
| `list_brains` | `scope?` | Registry listing with record counts, sizes, last activity, encoder, health. |
| `create_brain` | `name, scope, parent?, encoder?` | Create + register. |
| `brain_status` | `brain` | Counts by kind, staleness histogram, dup-cluster estimate, index state, journal lag, last maintenance runs. |
| `export_brain` | `brain, format: jsonl\|markdown\|both, since?` | §11.2. Returns export path + manifest hash. |
| `import_brain` | `path, mode: merge\|replace, re_embed?` | §11.3. |
| `maintenance` | `brain, op: candidates\|apply, task: prune\|dedupe\|reground\|synthesize\|deepen, payload?` | The mechanics half of lifecycle jobs (§12.3): server computes candidates; agents judge; server applies. |

All tools return structured JSON content; errors use MCP error codes with machine-readable
`reason` fields so agent retries can be conditional.

---

## 10. Skills — `/learn`, `/think`, `/relearn`, `/forget`

Skills are thin, judgment-bearing wrappers over the tools. They live in the braind repo under
`skills/` and install as a Claude Code plugin (the daemon is agent-agnostic; any MCP client gets
the tools without the skills).

| Skill | What the skill layer adds beyond the raw tool |
|-------|-----------------------------------------------|
| `/learn [brain] <insight>` | Distill the conversation into a *durable* record: proper title, atomic body, evidence extraction (file refs with commit SHAs, commands with expected output, URLs), kind classification, tag suggestion. Calls `think` first to detect "we already know this" → routes to `/relearn` instead. |
| `/think <question>` | Query formulation (expand the user's phrasing into retrieval-friendly text), call `think`, then synthesize an answer *from the returned records* with citations (`[perf:01J9Z…]`), flagging stale/low-confidence records instead of silently trusting them. |
| `/relearn <id \| description> <correction>` | Locate the target record (by id or via `think`), decide patch vs supersede (rule of thumb: meaning changed → supersede; wording/metadata → patch), update evidence, and note *why* in the journal `reason` field. |
| `/forget <id \| description>` | Locate, confirm with the user when matched by description rather than id (destructive-ish), tombstone by default, explain that journal history is retained. |
| `/brains [attach\|detach\|status\|list] …` | Attachment management + human-readable status. `/brains attach auto` = §8.2 discovery. |
| `/brain-export`, `/brain-import` | Wrappers over export/import with sensible defaults (export `both`, into the repo's `.braind/exports/` for project brains so knowledge rides along in git). |

Skill frontmatter marks `learn`/`relearn`/`forget` as requiring the `brains` MCP server, and each
SKILL.md documents the session-inheritance env var so subagent-spawning flows keep working.

---

## 11. Journal, export, import, and cross-host portability

### 11.1 Journal (the audit trail and WAL)

Every mutation appends one entry to `~/.braind/brains/<name>/journal/YYYY-MM.jsonl` *before* the
Lance commit (§6). Entries are self-contained — they carry the full record state, not a diff —
so the journal alone can rebuild a brain (vectors are recomputable).

```jsonc
{"seq": 4102, "ts": "2026-07-01T12:00:00Z", "op": "learn",     // learn|relearn|forget|purge|merge|compact|import|access
 "actor": {"host": "devbox-1", "session": "s_…", "agent": "claude-code", "job": null},
 "reason": "benchmarked fan-in variants for issue #42",
 "record": { …full record state after the op… },
 "prev_hash": "sha256:…", "hash": "sha256:…"}                   // hash chain → tamper-evident
```

`seq` is per-brain monotonic; the hash chain makes the audit log tamper-evident. `access` entries
(read telemetry) go to a separate `access-YYYY-MM.jsonl` to keep the mutation log clean.

### 11.2 Export

`export_brain` produces a self-contained directory:

```
exports/perf-2026-07-01T12-00-00Z/
├── manifest.json      # brain name/scope, record count, encoder {model, revision, dim},
│                      # reranker id, journal head {seq, hash}, schema_version, host
├── records.jsonl      # every live record, full fidelity (no vectors — recomputable)
├── tombstones.jsonl   # tombstoned records (ids + final state), for merge correctness
├── journal/           # copied journal segments (complete history)
└── markdown/          # human-auditable view, one file per tag-cluster:
    ├── INDEX.md       # TOC with counts, staleness, confidence summaries
    └── tokio-channels.md   # records rendered as sections: title, body, evidence,
                            # confidence/verified badges, supersede-chain links
```

JSONL is the machine round-trip format; Markdown is the human audit surface (diff-able in PRs when
project brains export into the repo). Both are regenerable from the journal, so "auditable files"
(requirement 5) never drift from the store.

### 11.3 Import

`import_brain { path, mode }`:

- **merge** (default): union by record id. Conflict (same id, both modified since common journal
  ancestor) → keep newer `updated_at`, journal a `merge` entry recording the loser's state. Supersede
  chains from both sides are preserved; tombstones win over live records (a forget anywhere is a
  forget everywhere).
- **replace**: wipe and rebuild from the export.
- If `manifest.encoder != brain.encoder`, records are **re-embedded** with the local encoder
  (vectors never travel; the text is the source of truth). Import is idempotent — re-importing the
  same export is a no-op — so periodic import from a synced directory is a valid replication mode.

### 11.4 Cross-host

v1 keeps it deliberately simple: export/import over whatever transport you already have (git for
project brains — exports committed in `.braind/exports/`; rsync/object storage for global brains).
Because imports are idempotent merges keyed on ULIDs and journals are append-only, this converges
(requirement 4's "eventual across brains/hosts"). A v2 `braind sync` peering protocol (journal-tail
shipping between daemons) slots in behind the same merge semantics without format changes.

---

## 12. Lifecycle management (background agentic workflows)

### 12.1 Principle: server does mechanics, agents do judgment

Deciding whether two records are truly duplicates, whether an insight still holds, or how to
summarize twenty episodes into one principle — that's LLM work. Finding candidates, holding locks,
and applying mutations atomically — that's server work. The `maintenance` tool (§9) is the
contract between them: `candidates` ops are cheap server-side queries (vector clusters, staleness
scans); `apply` ops are validated batch mutations, journaled with `actor.job` set.

### 12.2 Scheduler

A tokio-based cron in `braind` (`[maintenance]` section of config) that **fast-spawns short-lived
agent processes on a cadence** (requirement 9) — default runner is `claude -p "<job prompt>"
--mcp-config <braind endpoint>` in headless mode; the runner command is configurable per job so
any agent CLI works. Guard rails: per-job wall-clock timeout, one concurrent job per brain
(jobs take the same write path as everyone else, so MRSW is preserved), scoped auth tokens (§7.2),
jitter to avoid thundering herds, and a `paused` flag per brain.

```toml
# ~/.braind/config.toml
[maintenance]
runner = ["claude", "-p", "{prompt}", "--permission-mode", "dontAsk"]

[maintenance.jobs.prune]      cadence = "daily",   brains = "*"
[maintenance.jobs.dedupe]     cadence = "daily",   brains = "*"
[maintenance.jobs.reground]   cadence = "weekly",  brains = "*"
[maintenance.jobs.synthesize] cadence = "weekly",  brains = "*"
[maintenance.jobs.deepen]     cadence = "daily",   brains = "active-projects"  # activity-gated
```

### 12.3 The five jobs

| Job | Candidates (server) | Judgment + apply (agent) |
|-----|---------------------|--------------------------|
| **prune** | Score = f(age, access_count, confidence, superseded, kind) below threshold; episodes older than TTL | Agent reviews the list, spares anything still load-bearing (e.g. referenced as evidence by other records), tombstones the rest with reasons |
| **dedupe** | Cosine-similarity clusters (≥ 0.92) among live records, per brain | Agent merges each cluster: best-of title/body, union of evidence/tags, `merge` op supersedes losers |
| **re-ground** | Records with `last_verified_at` older than policy, ordered by access_count | Agent re-checks evidence: do the file refs still exist at those paths (at current HEAD)? do commands still produce expected output? do URLs still say what we claimed? → refresh `last_verified_at` + restore confidence, or decay confidence / `relearn` with corrections |
| **synthesize/compact** | Cohesive clusters of old `episode`/`fact` records (topic clusters with low recent access) | Agent writes one consolidated `insight`/`procedure` record superseding the members; originals tombstone; journal keeps everything |
| **deepen** | For **active** projects (commit/attach activity in last N days): high-access, low-coverage topics — frequent `think` queries whose top score was weak (server keeps a query-miss log) | Agent researches the gap (codebase + web), then `learn`s new well-evidenced records |

Confidence decays passively (e.g. half-life 90 days for `fact`/`insight`, none for `decision`)
so unverified knowledge sinks in ranking (§5.3) even between re-ground runs — staleness degrades
gracefully rather than lying confidently.

### 12.4 Reporting

Each job run journals a summary entry and emits a Markdown report to
`~/.braind/brains/<name>/reports/`, so `/brains status` can show "what maintenance did last night"
and the human can audit (and revert via journal) anything a background agent decided.

---

## 13. Implementation plan

### 13.1 Workspace layout (Rust)

```
braind/
├── Cargo.toml                 # workspace
├── crates/
│   ├── brain-core/            # types, registry, Lance store, locks, journal, export/import
│   ├── brain-embed/           # Embedder/LateInteractionEncoder traits; candle impls (bge-m3,
│   │                          #   answerai-colbert); hf-hub download/pinning; batching service;
│   │                          #   MaxSim kernel; optional `ort` fallback feature
│   ├── braind/                # daemon binary: axum + rmcp streamable HTTP, session registry,
│   │                          #   MCP tools, scheduler, auth, metrics
│   └── brainctl/              # CLI: start/stop/status, brain CRUD, export/import, journal tools
├── skills/                    # Claude Code plugin: learn/think/relearn/forget/brains/…
├── .claude-plugin/plugin.json
└── docs/DESIGN.md             # this document
```

Key dependencies: `lancedb`, `arrow`, `candle-core`/`candle-transformers`/`tokenizers`, `hf-hub`,
`rmcp`, `axum`, `tokio`, `serde`, `ulid`, `tracing` (+ `ort`/`fastembed` behind a feature).

### 13.2 Security & ops notes

- Bind 127.0.0.1 only by default; bearer token (§7.2); 0600 perms on data dirs.
- Model downloads honor the host proxy; models are pinned by revision hash in config (supply-chain
  hygiene) and verified against hf-hub etags.
- `tracing` structured logs; Prometheus `/metrics`: per-tool latency, embed queue depth, per-brain
  sizes, maintenance outcomes.
- Failure degradation: if models fail to load, `braind` still serves — `learn` queues records
  with `vector = null` for later backfill, `think` falls back to BM25-style full-text search over
  Lance (marked `degraded: true` in responses) — a broken GPU never blocks writing knowledge down.

### 13.3 Milestones

| | Scope | Exit criteria |
|--|-------|---------------|
| **M0** | Workspace scaffold, `brain-core` with Lance store + journal + RwLock registry, `brainctl` create/list | round-trip a record via `brainctl`; journal replay rebuilds a brain |
| **M1** | `brain-embed`: bge-m3 dense on candle, ANN search; `braind` MCP server with `learn`/`think`/`attach`/`list` | two concurrent Claude Code sessions share one brain with read-after-write |
| **M2** | ColBERT write-time token vectors + MaxSim rerank; `relearn`/`forget`; skills plugin | rerank measurably beats dense-only on a small eval set (evals/ checked in) |
| **M3** | Export/import (jsonl + markdown), merge semantics, project-brain auto-discovery, `.braind/` repo config | export on host A → import on host B → identical `think` results |
| **M4** | Scheduler + all five maintenance jobs with `claude -p` runner; reports | a week of nightly runs on a real project brain produces sensible prune/dedupe/synthesize journal entries |
| **M5** | Hardening: degraded modes, metrics, scoped tokens, docs; (stretch) `braind sync` peering | — |

---

## 14. Open questions

1. **Chunking policy** — fixed ~512-token chunks vs. structure-aware (markdown headings/code
   blocks). Start fixed; revisit after M2 evals.
2. **Lance multivector maturity in the Rust crate** — native MaxSim vs. in-process kernel. The
   design works either way (§5.3); pick at M2 based on what the crate ships.
3. **Query-miss log retention** for the deepen job — how much query telemetry to keep, and does it
   need the same audit treatment as knowledge (leaning: no, 30-day rolling window, not exported).
4. **Plan-brain end-of-life** — when a plan completes, auto-synthesize its brain into the project
   brain and archive? (Leaning yes: a `plan-complete` maintenance task in v1.1.)
5. **Cross-encoder final stage** — a third rerank stage (e.g. bge-reranker) for `think` when n is
   small. Deferred: ColBERT should be sufficient; measure at M2.

---

## Appendix A — example flows

**Learn (session S, project brain P attached, write target P):**
`/learn "tokio mpsc beats crossbeam for fan-in …"` → skill distills record + evidence →
`learn` tool → dup check (cosine 0.97 vs r_123? → suggest relearn) → embed dense + ColBERT →
journal append (fsync) → Lance insert commit → `{id}` → skill confirms with citation.

**Think (same host, different session, immediately after):**
`/think "which channel impl for fan-in?"` → `think` → ANN top-100 across {P, perf} → MaxSim
rerank → the record just learned ranks #1 (read-after-write, §6) → skill answers with
`[project:01J9Z…]` citation and its evidence.

**Nightly re-ground on `perf`:** scheduler → `claude -p` with scoped token → `maintenance
{op: candidates, task: reground}` → agent re-runs `cargo bench --bench fanin` → 1.4x still holds →
`maintenance {op: apply}` refreshes `last_verified_at`, restores confidence 0.9 → journal + report.
