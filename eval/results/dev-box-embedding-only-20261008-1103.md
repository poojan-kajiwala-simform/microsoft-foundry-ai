# Local RAG evaluation: dev-box-embedding-only

- Date: 2026-10-08 11:00 UTC
- Machine: Microsoft Windows NT 10.0.26200.0, 12 logical cores, AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD, 15 GB RAM
- Execution providers registered: False
- Chat model: not loaded, embedding model: qwen3-embedding-0.6b (variant `qwen3-embedding-0.6b-generic-cpu:1`)
- Cold start (this run): embedding load 2.1s (models were already cached)
- Corpus: 3 files; golden set: 12 answerable + 4 unanswerable questions
- Config: TopK=4, MinSimilarity=0.4, chunk=300w/40w overlap, context budget=1500 tokens

## Ingestion

| Variant | Files | Chunks | Total s | Embedding s | Chunks/s | Re-run (no changes) s |
|---|---|---|---|---|---|---|
| fixed 300w + vector | 3 | 3 | 10.6 | 10.4 | 0.3 | 0.03 (unchanged: 3) |
| fixed 60w + vector | 3 | 16 | 25.3 | 25.3 | 0.6 | 0.00 (unchanged: 3) |
| structured 300w + vector | 3 | 15 | 24.2 | 24.1 | 0.6 | 0.00 (unchanged: 3) |
| structured 300w + hybrid | 3 | 15 | 24.7 | 24.6 | 0.6 | 0.00 (unchanged: 3) |

Process memory after ingestion: working set 880 MB (models run in-process).

## Retrieval quality

Hit@K = a chunk from the expected source containing the evidence phrase is in the top 4; Hit@1 requires it to be ranked first (the stricter test on a corpus this small). MRR uses the first such chunk. Margin > 0 means a single similarity threshold can separate answerable from unanswerable questions.

| Variant | Hit@1 | Hit@K | MRR | Margin (min answerable − max unanswerable) | Answerable: top similarity min / median | Unanswerable: top similarity median / max | Abstention accuracy @ threshold | Best threshold on this set |
|---|---|---|---|---|---|---|---|---|
| fixed 300w + vector | 12/12 | 12/12 (100%) | 1.00 | -0.03 | 0.35 / 0.47 | 0.24 / 0.39 | 14/16 | 0.35 |
| fixed 60w + vector | 12/12 | 12/12 (100%) | 1.00 | +0.11 | 0.46 / 0.71 | 0.29 / 0.36 | 16/16 | 0.46 |
| structured 300w + vector | 12/12 | 12/12 (100%) | 1.00 | +0.04 | 0.43 / 0.69 | 0.23 / 0.39 | 16/16 | 0.43 |
| structured 300w + hybrid | 12/12 | 12/12 (100%) | 1.00 | +0.04 | 0.43 / 0.69 | 0.23 / 0.39 | 16/16 | 0.43 |

Misses (best variant):

