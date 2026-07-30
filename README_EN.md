# kazakh-granite-retriever

*Русская версия — [README.md](README.md).*

**🤗 Model on HuggingFace: [`Tim2190/granite-278m-kk`](https://huggingface.co/Tim2190/granite-278m-kk)** (fp16, ~556 MB).
Current revision is **v2** (morphology-hardened); the previous one is available as `revision="v1"`.

**Granite-278m-kk** — a compact (278M) embedding model for Kazakh search and RAG,
fine-tuned from `ibm-granite/granite-embedding-278m-multilingual` (R1). Kazakh is not
in Granite's official language list — this model shows targeted fine-tuning closes
that gap in practice.

> **📄 Full report — [REPORT_EN.md](REPORT_EN.md)** (method, all tables, significance).

## Summary (honest, by significance)

All comparisons run at seq 512, using the benchmark's own harness and metrics;
significance is paired bootstrap (10k).

- **Significantly improves the base Granite-R1** both in-domain (Wiki: 0.672 → **0.751**
  dense / **0.813** hybrid, p<0.001) and on an independent OOD set (speeches: 0.430 →
  **0.529**, every tier p<0.05).
- **On par with the specialized kazakh-e5** — a statistical tie on ALL (0.751 vs 0.747,
  p=0.42). No win here, and we will not claim one.
- **v2 is significantly stronger than v1 on morphology** (inflected 0.752 → 0.792,
  p=0.002) — the effect of hard negatives mined with a Kazakh stemmer. This **closes
  the one place kazakh-e5 still led** (morphology: was p=0.001 in its favor → now p=0.06,
  a tie).

**Primary benchmark (Wikipedia), nDCG@10 @512:**

| slice | zero-shot | v2 dense | v2 ⊕ BM25 | v2 vs kazakh-e5 (Δ, p) |
|---|---|---|---|---|
| **ALL** | 0.672 | **0.751** | **0.813** | +0.004, p=0.42 (tie) |
| inflected | 0.791 | 0.792 | 0.822 | −0.044, p=0.06 (tie) |
| natural | 0.923 | 0.928 | 0.888 | +0.019, p=0.21 |
| vocabulary-gap | 0.303 | 0.534 | 0.728 | +0.037, p=0.17 |

Where the model sits (we have no paired test across third-party systems, so this is
approximate): the hybrid **0.813** is on par with kazakh-e5 ⊕ BM25 (~0.808) and above
e5-base (~0.785); strong general multilingual models (bge-m3 ~0.866, jina-v3 ~0.821)
remain ahead — a target for further work.

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
| synthetic data (57K pairs) | `scripts/generate_synthetic.py` |
| hard negatives, BM25 + Kazakh stemmer | `scripts/mine_hard_negatives.py` |
| build train file (id → texts) | `scripts/build_train_v2.py` |
| training (T4, CachedMNRL, seq 512) | `scripts/train.py` |
| evaluation + hybrid with BM25(stemmer) | `scripts/eval.py` |
| evaluation (OOD: speeches) | `scripts/eval_ood.py` |

Ready-to-run Colab/Kaggle notebooks are in `notebooks/`. The Kazakh stemmer (used for
hard negatives and the hybrid's BM25 channel, key `KAZAKH_STEMMER_KEY`) is available at
[qaz-api.vercel.app](https://qaz-api.vercel.app/).

**v2 was trained on:** 57,369 synthetic pairs (Kazakh Wikipedia) + 1 hard negative per
pair (mined via BM25 with a **Kazakh stemmer**) + 3,733 KazQAD gold (with its rel=0
negatives) = 61,102 training examples. 278m, CachedMNRL, 2 epochs, lr 1e-5,
**max_seq_len 512**, Kaggle T4. What differs from v1: the stemmer negatives (morphology)
and training at seq 512.

## Data and license

`data/synthetic_pairs.jsonl` — 57,369 pairs (after anti-leak), synthetic over **KazQAD**
passages (Kazakh Wikipedia, **CC BY-SA 4.0**, attribution to KazQAD). Compact negative
artifacts — `data/synthetic_pairs.hn.ids.jsonl`, KazQAD gold — `data/kazqad_pairs.dedup.jsonl`.
Full results in `results/`.

Model: [`Tim2190/granite-278m-kk`](https://huggingface.co/Tim2190/granite-278m-kk)
(fp16, ~556 MB; v2 on `main`, v1 under the `v1` tag).
