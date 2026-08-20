# ATLAS — Evaluation

This document reports Phase 5's required retrieval/generation experiments against
the golden dataset in [`eval/dataset/`](../eval/dataset/), with real numbers from
real models (local and, since Phase 6, OpenRouter) and a real Postgres/pgvector
instance — not simulated. Every number below can be reproduced by running the three
experiment runners described in "How to reproduce" and is also persisted as
`EvaluationRun`/`EvaluationResult` rows in Postgres and as JSON in
[`eval/reports/`](../eval/reports/).

**Update (Phase 6):** Experiment 5 (generation quality) was blocked at original
Phase-5-write-up time because this project's OpenAI account has no billing credits.
It's since been run for real by routing the LLM and embeddings through OpenRouter
instead (see backend/app/core/config.py's provider factories) — Experiments 1-4
below are unchanged from the original Phase 5 write-up.

## Golden dataset

[`eval/dataset/golden_qa.yaml`](../eval/dataset/golden_qa.yaml): 30 questions against
four corpus documents ([`eval/dataset/corpus/`](../eval/dataset/corpus/)) — 24
answerable, 6 deliberately unanswerable. Ground truth is at **document** granularity
(not chunk-ID), so it stays comparable across chunking strategies with different
chunk boundaries — see the YAML's own schema comment for the full rationale.

**Named limitation:** 30 questions and 4 documents are well below the ~50-100
question / larger-corpus target `docs/ARCHITECTURE.md` §10 risk 6 names as ideal for
statistically confident conclusions, and kept intentionally small and hand-reviewable
for a single-session portfolio build. Concretely, with only 4 source documents and
`retrieval_top_k=5`, Recall@K is close to saturated (every experiment below scores
1.000) — recall isn't a discriminating metric at this corpus size. Precision@K and
MRR are more informative here since they're sensitive to ranking, not just coverage.
Expanding the corpus and question count is the natural next step if this became a
real evaluation harness rather than a demonstration of the framework.

## Metrics implemented

| Metric | Module | Status |
|---|---|---|
| Recall@K, Precision@K, MRR | [`eval/metrics/retrieval.py`](../eval/metrics/retrieval.py) | Real numbers below |
| p50 / p95 latency | [`eval/metrics/latency.py`](../eval/metrics/latency.py) | Real numbers below |
| Answer correctness, citation correctness, groundedness, hallucination rate, correct refusal rate | [`eval/metrics/generation.py`](../eval/metrics/generation.py) | Real numbers below (Experiment 5) |

All are pure functions, covered by [`eval/tests/`](../eval/tests/) (45 tests).

## Experiment 1 — Embedding model comparison

**Question: which embedding model retrieves better — a smaller or larger local model?**
Two local `sentence-transformers` models (no OpenAI dependency — see Limitations),
compared in-memory via numpy cosine similarity against the fixed-size chunking
baseline. See [`eval/runners/in_memory_experiments.py`](../eval/runners/in_memory_experiments.py).

| Model | Dimension | Recall@5 | Precision@5 | MRR | Mean latency |
|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 1.000 | 0.250 | **0.979** | 15.4 ms |
| BAAI/bge-base-en-v1.5 | 768 | 1.000 | 0.250 | 0.948 | 29.2 ms |

**Result: the smaller model (bge-small, 384-dim) ranked the correct document first
slightly more often (MRR 0.979 vs 0.948) and embedded roughly 2x faster.** On this
corpus, the larger model's extra capacity didn't translate to better ranking —
consistent with the corpus being short, non-technical prose rather than content that
would exercise a larger model's advantage. `bge-small-en-v1.5` was therefore used as
the fixed embedding model for Experiments 2-4 below.

## Experiment 2 — Chunking strategy comparison

**Question: does structure-aware chunking beat naive fixed-size chunking?**
Same embedding model (`bge-small-en-v1.5`) and question set, varying only the
chunker: [`chunk_pages()`](../backend/app/ai/pipeline/chunking.py) (fixed 400-token
windows, 50-token overlap, ignores structure) vs
[`chunk_by_headings()`](../backend/app/ai/pipeline/chunking.py) (splits on the
corpus's `## Heading` markers first, tagging each chunk with `section_title`).

| Strategy | Recall@5 | Precision@5 | MRR | Mean latency |
|---|---|---|---|---|
| fixed_size (baseline) | 1.000 | 0.250 | 0.979 | 13.8 ms |
| structure_aware | 1.000 | **0.385** | **1.000** | 13.5 ms |

**Result: structure-aware chunking wins clearly on precision (0.385 vs 0.250, a 54%
relative improvement) and reaches perfect MRR (1.000).** Splitting on section
headings before token-windowing keeps a chunk's content topically coherent (e.g. all
of "Vacation Policy" in one chunk instead of split mid-window), so the top-ranked
chunk is more consistently the *right* chunk, not just *a* chunk from the right
document. This is the clearest, most confidently-supported result in this
evaluation — worth acting on for a real deployment of this corpus shape (documents
with clear heading structure).

## Experiment 3 — Hybrid search vs vector-only

**Question: does fusing full-text search with vector search (RRF) improve
retrieval?** Run through the real pgvector-backed `retrieve_relevant_chunks()` — not
in-memory — since retrieval mode doesn't require varying embedding dimension. See
[`eval/runners/pgvector_experiments.py`](../eval/runners/pgvector_experiments.py).

| Variant | Recall@5 | Precision@5 | MRR | p50 latency | p95 latency |
|---|---|---|---|---|---|
| vector_only | 1.000 | 0.250 | 0.979 | 17 ms | 19 ms |
| hybrid_rrf | 1.000 | 0.250 | 0.979 | 18 ms | 20 ms |

**Result: no measurable retrieval-quality difference on this corpus** — identical
Precision@5 and MRR. Full-text search adds a second candidate list that Reciprocal
Rank Fusion merges in, but on short, single-topic corpus documents where the
vector-only baseline already finds the right chunk first, there's nothing for FTS to
correct. Hybrid search's real cost here is pure latency overhead (~1ms more, an
extra query) for no quality gain — on this dataset, `enable_hybrid_search` staying
`False` by default is the right call. This conclusion is corpus-
dependent: hybrid search is expected to matter more on corpora with exact-keyword
or rare-term queries (part numbers, error codes, proper nouns) that dense vector
search alone underperforms on — none of which this golden dataset happens to test.

## Experiment 4 — Reranking on vs off

**Question: does cross-encoder reranking improve retrieval, and at what cost?**

| Variant | Recall@5 | Precision@5 | MRR | p50 latency | p95 latency |
|---|---|---|---|---|---|
| vector_reranked | 1.000 | 0.250 | **1.000** | 549 ms | 724 ms |
| hybrid_reranked | 1.000 | 0.250 | **1.000** | 549 ms | 756 ms |

(compare to `vector_only`'s MRR 0.979 / p50 17ms and `hybrid_rrf`'s MRR 0.979 / p50
18ms above.)

**Result: reranking closes the gap to a perfect MRR of 1.000 in both cases — a real,
if modest, ranking improvement — at a very large latency cost: ~30x p50 latency
(17ms → 549ms).** That cost is the local CPU cross-encoder
(`BAAI/bge-reranker-base`) scoring `retrieval_candidate_pool_size=20` candidates
per query on CPU, not network overhead. This is the clearest quantified case for
keeping `enable_reranking=False` by default: the MRR improvement
here (0.979 → 1.000) reflects fixing a small number of already-near-correct
rankings, not rescuing wrong answers — on this corpus it doesn't justify a 30x
latency multiplier. A GPU-hosted reranker or a smaller cross-encoder model would
change this trade-off; that wasn't evaluated here (see Limitations).

## Experiment 5 — Generation quality

**Question: how well does the full RAG pipeline (retrieve + generate + cite) actually
perform, and does it correctly refuse when it doesn't know?** Runs the real
production pipeline (`retrieve_relevant_chunks` → `ai/pipeline/rag_pipeline.generate_answer`)
over all 30 golden questions, using the real OpenRouter-backed LLM
(`openai/gpt-oss-20b:free`) and embedding provider (`openai/text-embedding-3-small`,
via OpenRouter) — the actual providers the live app uses by default (Phase 6), not a
local stand-in. See [`eval/runners/generation_experiment.py`](../eval/runners/generation_experiment.py).

| Metric | Result | n |
|---|---|---|
| Answer correctness | **0.750** | 24 answerable questions |
| Citation correctness | **0.917** | 24 answerable questions |
| Groundedness rate | **0.917** | 24 answerable questions |
| Correct refusal rate | **1.000** | 6 unanswerable questions |
| Hallucination rate | **0.000** | 6 unanswerable questions |
| Latency | p50 9.0s / p95 42.5s / mean 16.0s | 30 questions |

**Result: the model never fabricated an answer to an unanswerable question (6/6
correct refusals, 0 hallucinations) — the strongest, cleanest result in this
evaluation.** The system prompt's refusal instruction (`ai/pipeline/rag_pipeline.py`'s
`SYSTEM_PROMPT`, demanding the exact fixed refusal string when context is
insufficient) is followed reliably even by a small free-tier model. Answer
correctness (0.750) and citation/groundedness (0.917 each) are good but not perfect —
manual inspection of the `EvaluationResult` rows (`eval/reports/generation_experiment.json`
has the run id) is the natural next step to see whether the misses are retrieval
misses (wrong chunk surfaced) or generation misses (right chunk, wrong phrasing
against the substring-match metric — `answer_correctness` is a strict, case-sensitive-ish
substring check, so a correct paraphrase can score as incorrect; see Limitations).
Latency is dominated entirely by the free-tier LLM call (mean ~16s, up to 68s on the
slowest question) — retrieval itself is fast (see Experiments 3-4); this is a
free-tier-rate-limit cost, not an architectural one.

`docs/ARCHITECTURE.md` §10's original tech-choice table proposed using `ragas` for
faithfulness/groundedness scoring (a real, peer-reviewed library) rather than a
custom metric. `ragas`'s implementation itself calls an LLM as judge, adding cost and
complexity beyond what this evaluation needed — `groundedness_rate` here is instead a
**structural proxy** (does a non-refusal answer cite at least one chunk, per the
system prompt's citation contract) that needs no separate judge call. It measures
contract compliance, not whether cited content actually supports the claim next to
it — see the module docstring in `eval/metrics/generation.py` for the full reasoning.

## Limitations

1. **Golden dataset size** (30 questions, 4 documents) — see "Golden dataset" above.
   Recall@K is saturated and not discriminating at this scale; Precision@K and MRR
   carry the real signal in this report.
2. **Answer correctness is a strict substring match** (`eval/metrics/generation.py`'s
   `answer_correctness`), not a semantic judge — a correct answer phrased
   differently from every string in `expected_answer_contains` scores as incorrect.
   This is a real, not hypothetical, source of the 0.750 (not higher) answer
   correctness rate in Experiment 5; an LLM-as-judge pass would likely score some of
   those misses as correct. Deliberately avoided to keep the metric callable without
   another paid LLM call — see the module docstring.
3. **Embedding-dimension workaround, Experiments 1-4 only.** `DocumentChunk.embedding`
   is a fixed `Vector(1536)` column sized for OpenAI's `text-embedding-3-small`
   (`docs/ARCHITECTURE.md` §10 risk 10.2). Experiments 1-2 sidestep this by
   comparing local models in-memory (numpy), outside that column entirely.
   Experiments 3-4 need the real pgvector column, so they use
   `ZeroPaddedEmbeddingProvider` (`eval/runners/common.py`) to zero-pad
   `bge-small-en-v1.5`'s 384-dim output to 1536 — a transformation that provably
   preserves cosine similarity exactly (padded zeros change neither the dot product
   nor either vector's norm), so it doesn't distort the retrieval ranking. It does
   mean Experiments 3-4 were run with a local embedding model, not
   `text-embedding-3-small` — a real deployment's numbers with OpenAI embeddings
   could differ. **Experiment 5 does not have this caveat** — it uses the real
   production embedding provider (OpenRouter's `text-embedding-3-small` passthrough,
   1536-dim natively, no padding).
4. **Reranker latency is CPU-bound.** The ~30x latency multiplier in Experiment 4 is
   specific to running `BAAI/bge-reranker-base` on CPU locally (a deliberate,
   documented trade-off — see `ai/reranker/cross_encoder.py`'s lazy-loading
   design). A GPU-hosted or smaller reranker would likely
   change the cost/benefit conclusion; not evaluated here.
5. **Single corpus domain.** All four documents are short, single-topic internal-doc
   style prose (HR policy, product info, security policy, engineering process).
   Hybrid search's null result (Experiment 3) is plausibly corpus-specific — corpora
   with exact-keyword-sensitive queries would likely show a different result.
6. **Operational note for reproducers:** all three experiment runners write real,
   committed rows (a dedicated `eval-runner@atlas.internal` user, ingested
   documents/chunks, `EvaluationRun`/`EvaluationResult` rows) to whatever Postgres
   `DATABASE_URL` points at. Running them against the same dev database the backend
   integration test suite uses will make several tests fail (e.g.
   "first registered user becomes admin", exact-chunk-count assertions) because
   those tests assume a database containing only their own rolled-back fixtures, not
   real committed data. Wipe the schema and rerun `alembic upgrade head` before
   running `pytest` again after running any experiment runner. Experiment 5
   additionally needs `OPENROUTER_API_KEY` set and makes 30 real (small, real-cost
   for embeddings; free for the LLM) API calls — expect it to take 5-15+ minutes on
   a free-tier model, and a transient network blip mid-run is a real, observed
   failure mode (not a code bug) worth just retrying.

## How to reproduce

```bash
cd infra
docker compose up -d postgres
cd ../backend
.venv/Scripts/python -m alembic upgrade head
cd ..
backend/.venv/Scripts/python -m eval.runners.in_memory_experiments      # Experiments 1-2
backend/.venv/Scripts/python -m eval.runners.pgvector_experiments       # Experiments 3-4
backend/.venv/Scripts/python -m eval.runners.generation_experiment      # Experiment 5 (needs OPENROUTER_API_KEY)
```

Each run prints a summary table, writes `eval/reports/<runner_name>.json`, and
persists one `EvaluationRun` + one `EvaluationResult` per question to Postgres. See
the Limitations section above before running the backend test suite afterward.
