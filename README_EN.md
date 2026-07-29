# kazakh-granite-retriever

*Русская версия — [README.md](README.md).*

**Granite-278m-kk** — a compact (278M) embedding model for Kazakh search and RAG,
fine-tuned from `ibm-granite/granite-embedding-278m-multilingual` (R1). Kazakh is not
in Granite's official language list — this model shows targeted fine-tuning closes
that gap in practice.

> **📄 Full report — [REPORT_EN.md](REPORT_EN.md)** (method, all tables, conclusions).

## Summary

- Beats the specialized Kazakh fine-tunes (kazakh-e5, kazembed-v5) on two independent
  domains.
- The BM25 hybrid (**0.814** nDCG@10) beats the kazakh-e5 ⊕ BM25 reference (0.808) and
  base e5-base (0.785) on the primary benchmark.
- Significantly improves the base Granite-R1 (0.672 → 0.752 dense / 0.814 hybrid).
- The gain is confirmed on an independent OOD domain (official speeches) — significant
  across all tiers.

**Primary benchmark (Wikipedia), nDCG@10 (ALL):**

| # | system | ALL |
|---|---|---|
| 1 | bge-m3 | 0.866 |
| 2 | jina-v3 | 0.821 |
| **3** | **Granite-278m-kk ⊕ BM25 (ours)** | **0.814** |
| 4 | kazakh-e5 ⊕ BM25 | 0.808 |
| 6 | multilingual-e5-base | 0.785 |
| **8** | **Granite-278m-kk (ours, dense)** | **0.752** |
| 9 | kazakh-e5 | 0.747 |

Top tier: the dense model beats the specialized kazakh-e5, and the hybrid beats its
BM25 combo and e5-base. Strong general models (bge-m3, jina-v3) are ahead — a target
for further work.

## Benchmarks & sources

Ratings and evaluation run on these benchmarks; our `eval.py` / `eval_ood.py`
**reuse their harness and metrics** so the numbers are comparable:

- **[Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)** —
  the primary benchmark (Kazakh Wikipedia, 300 queries, 8,370 passages). Source of the
  comparative leaderboard and the eval harness: `src/retrieval` (DenseIndex, BM25 +
  Kazakh stemmer), `src/eval` (metrics, paired bootstrap) — imported directly by our scripts.
- **[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)** —
  the independent OOD benchmark (official speeches, akorda.kz / nazarbayev.kz).
- **[KazQAD](https://github.com/IS2AI/KazQAD)** — source of the training passages
  (Kazakh Wikipedia, CC BY-SA 4.0).

## Pipeline

| step | script |
|---|---|
| base selection (zero-shot) | `scripts/zeroshot_107m.py` |
| KazQAD → pairs + KazQAD negatives (rel=0) | `scripts/prepare_data.py` |
| anti-leak vs benchmark | `scripts/check_overlap.py` |
| synthetic data (40K pairs) | `scripts/generate_synthetic.py` |
| training (T4, CachedMNRL) — synthetic + KazQAD gold | `scripts/train.py` |
| evaluation + hybrid with BM25(Kazakh stemmer) | `scripts/eval.py` |
| evaluation (OOD: speeches) | `scripts/eval_ood.py` |
| _(ablation, not in final)_ BM25 hard-neg mining | `scripts/mine_hard_negatives.py` |

The final model is trained on **synthetic (40K) + KazQAD gold** (with its `rel=0`
negatives). The Kazakh stemmer is used **only in the hybrid's BM25 channel** at
evaluation. BM25 hard-negative mining (`mine_hard_negatives.py`) was explored as an
ablation — it gave no improvement and is not in the final model.

## Data and license

`data/synthetic_pairs.jsonl` — 40,084 pairs, synthetic over **KazQAD** passages
(Kazakh Wikipedia, **CC BY-SA 4.0**, attribution to KazQAD). Full results in `results/`.

The model will be published on HuggingFace (link to be added).
