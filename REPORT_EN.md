# Granite-278m-kk — Kazakh retrieval with a fine-tuned IBM Granite

*Русская версия — [REPORT.md](REPORT.md).*

A compact (278M) embedding model for Kazakh search and RAG, fine-tuned from
`ibm-granite/granite-embedding-278m-multilingual` (R1). Kazakh is **not** in Granite's
official language list — this model shows targeted fine-tuning closes that gap in
practice: it significantly improves the base, generalizes to a different domain, and
runs on par with specialized Kazakh models.

All numbers are at **seq 512**, using the benchmark's own harness and metrics;
significance is paired bootstrap (10,000 resamples). The current model is **v2**;
the previous one is available as `revision="v1"`.

---

## At a glance (honest)

- **Significantly improves the base Granite-R1** — Wiki 0.672 → **0.751** (dense) /
  **0.813** (hybrid), p<0.001; OOD (speeches) 0.430 → **0.529**, every tier p<0.05.
- **On par with the specialized kazakh-e5** — a statistical tie on ALL (0.751 vs 0.747,
  p=0.42). No win.
- **v2 is significantly stronger than v1 on morphology** (inflected 0.752 → 0.792,
  p=0.002) — the effect of stemmer-mined hard negatives. This **closes the one place
  kazakh-e5 still led** (morphology: was p=0.001 in its favor → now p=0.06, a tie).
- **The gain generalizes** to an independent OOD domain, including the genuinely
  semantic tiers.
- Compact (278M), trained on a single Kaggle T4, fully reproducible.

## Where the model stands (honestly)

Primary benchmark [Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
(Wikipedia, 300 queries, 8,370 passages), nDCG@10 (ALL), @512. We have no paired test
across third-party systems, so their numbers are a reference, not a strict ranking:

| system | ALL |
|---|---|
| bge-m3 | ~0.866 |
| jina-v3 | ~0.821 |
| **Granite-278m-kk ⊕ BM25 (our hybrid)** | **0.813** |
| kazakh-e5 ⊕ BM25 | ~0.808 |
| multilingual-e5-base | ~0.785 |
| BM25 + Kazakh stemmer | 0.757 |
| **Granite-278m-kk (ours, dense)** | **0.751** |
| kazakh-e5 (specialized) | 0.747 |
| Granite-R1 278m (base, zero-shot) | 0.672 |

The model sits in the top tier: the hybrid is on par with kazakh-e5 ⊕ BM25, and the
dense model is level with kazakh-e5 (a paired tie). Strong general models (bge-m3,
jina-v3) remain ahead — a target for further work. **We do NOT claim to beat kazakh-e5**
— on ALL it is a tie.

## Benchmarks & sources

- **[Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)** —
  the primary benchmark (Kazakh Wikipedia). Source of the comparative leaderboard and the
  eval harness: `src/retrieval` (DenseIndex, BM25 + Kazakh stemmer), `src/eval` (metrics,
  paired bootstrap). Our `eval.py` / `eval_ood.py` import it directly.
- **[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)** —
  the independent OOD benchmark (official speeches, akorda.kz / nazarbayev.kz).
- **[KazQAD](https://github.com/IS2AI/KazQAD)** — training passages (Kazakh Wikipedia,
  CC BY-SA 4.0).

## How it was built

1. **Base selection (`zeroshot_107m.py`).** Zero-shot of all Granite models. Flagship —
   **278m R1** (the strongest zero-shot Granite, 0.672).
2. **Data (`prepare_data.py`).** KazQAD: 825K passages + labeled triples → 3,893
   query→gold pairs + hard negatives (rel=0).
3. **Anti-leak (`check_overlap.py`).** Dedup of train against the benchmark by title +
   near-dup + article_id.
4. **Synthetic data (`generate_synthetic.py`).** Question generation (Gemini) over Kazakh
   Wikipedia passages, balanced by type, with strict anti-leak → **57,369 pairs** (after
   dedup) from ~19K articles.
5. **Hard negatives (`mine_hard_negatives.py`).** For each query, top BM25 with a **Kazakh
   stemmer** minus the gold → morphological decoys (1 per pair). This step is what
   distinguishes v2 from v1.
6. **Train-file assembly (`build_train_v2.py`).** id → texts from corpus + KazQAD gold =
   61,102 training examples.
7. **Training (`train.py`).** sentence-transformers, CachedMNRL, 278m, 2 epochs, lr 1e-5,
   **max_seq_len 512**, fp16 save, Kaggle T4 (~7.5 h on 2×T4).
8. **Evaluation (`eval.py`, `eval_ood.py`).** Same harness and metrics, paired bootstrap
   (10k), strictly at seq 512. Hybrid — RRF(dense, BM25 + Kazakh stemmer).

## Results — primary benchmark (Wikipedia), @512

**Zero-shot vs v2 (dense, nDCG@10):**

| slice | zero-shot | v2 | Δ | p |
|---|---|---|---|---|
| **ALL** | 0.672 | **0.751** | +0.079 | **<0.001** |
| natural | 0.923 | 0.928 | +0.004 | 0.384 |
| inflected | 0.791 | 0.792 | +0.002 | 0.475 |
| vocabulary-gap | 0.303 | 0.534 | +0.231 | **<0.001** |

**Hybrid with BM25 (Kazakh stemmer):**

| slice | dense | BM25(kaz) | **hybrid** |
|---|---|---|---|
| **ALL** | 0.751 | 0.757 | **0.813** |
| natural | 0.928 | 0.772 | 0.888 |
| inflected | 0.792 | 0.736 | 0.822 |
| vocabulary-gap | 0.534 | 0.764 | 0.728 |

**v2 vs kazakh-e5 (paired bootstrap):**

| slice | v2 | kazakh-e5 | Δ | p |
|---|---|---|---|---|
| **ALL** | 0.751 | 0.747 | +0.004 | 0.419 (tie) |
| inflected | 0.792 | 0.836 | −0.044 | 0.063 (tie) |
| natural | 0.928 | 0.909 | +0.019 | 0.213 |
| vocabulary-gap | 0.534 | 0.497 | +0.037 | 0.168 |

ALL is a tie. Importantly: for v1, kazakh-e5 beat us on inflected **significantly**
(p=0.001); for v2 that gap is **no longer significant** (p=0.063) — the specialized
model's morphology advantage is closed.

**v2 vs v1 (what the stemmer-negative round bought):**

| slice | v2 | v1 | Δ | p |
|---|---|---|---|---|
| ALL | 0.751 | 0.742 | +0.009 | 0.084 |
| **inflected** | **0.792** | **0.752** | **+0.040** | **0.002** |
| natural | 0.928 | 0.921 | +0.007 | 0.106 |
| vocabulary-gap | 0.534 | 0.553 | −0.019 | 0.055 |

Only morphology improved **significantly** (inflected, p=0.002) — exactly what the
stemmer negatives targeted. On ALL the gain is nominal but not significant (p=0.084).
vocab-gap dipped slightly (p=0.055, borderline) — an honest trade-off: hardening
morphology cost a little on discriminating close entities.

## Results — independent validation (OOD: speeches, Akorda), @512

Second benchmark [RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ):
471 passages from akorda.kz / nazarbayev.kz speeches — a **different domain**, not seen
in training.

| tier | zero-shot | v2 | Δ | p |
|---|---|---|---|---|
| **ALL** | 0.430 | 0.529 | +0.099 | <0.001 |
| factoid | 0.548 | 0.680 | +0.132 | <0.001 |
| paraphrase | 0.406 | 0.503 | +0.097 | 0.004 |
| low_overlap | 0.339 | 0.405 | +0.066 | 0.009 |

Hybrid on Akorda — 0.554. The gain holds **on a different domain** and on the genuinely
semantic tiers (paraphrase, low_overlap) — a real skill, not overfitting to benchmark #1.
Versus v1 (dense 0.507 / hybrid 0.552), v2 is nominally a touch higher (0.529 / 0.554);
there is no paired OOD test, so we say "no worse", nothing stronger.

## Future work

The experiment succeeded: fine-tuning turned an officially "unsupported" Granite into a
competitive Kazakh retriever that runs level with specialized models and significantly
improves the base on two domains. Levers to push further:

1. **More hard negatives per pair** (v2 uses 1; 2–4 may push morphology/ALL further).
2. **More synthetic data** (57K → 100K+; the KazQAD corpus of 825K passages allows it).
3. **Recover vocab-gap** — rebalance negatives to avoid losing on close-entity discrimination.
4. **Hybrid as a single component** — wrap Granite-ft + BM25(stemmer) + RRF into one `.search()`.
5. **Analyze the bge-m3 / jina-v3 gap** — what gives strong general models their edge.

## Usage

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("Tim2190/granite-278m-kk")   # 🤗 fp16, ~556 MB (v2)
emb = m.encode(["Балқаш көлі қайда орналасқан?"])     # no special prefixes
# previous version: SentenceTransformer("Tim2190/granite-278m-kk", revision="v1")
```

For the strongest setup, use the BM25 hybrid (RRF); see `eval.py --hybrid` and `scripts/`.

## Data, license, reproducibility

- **Training data:** `data/synthetic_pairs.jsonl` (57,369 pairs) + hard negatives
  (`data/synthetic_pairs.hn.ids.jsonl`) + KazQAD gold (`data/kazqad_pairs.dedup.jsonl`) —
  synthetic over KazQAD passages (Kazakh Wikipedia, **CC BY-SA 4.0**, attribution to KazQAD).
- **Scripts:** `zeroshot_107m.py`, `prepare_data.py`, `check_overlap.py`,
  `generate_synthetic.py`, `mine_hard_negatives.py`, `build_train_v2.py`, `train.py`,
  `eval.py`, `eval_ood.py`. Run notebooks — `notebooks/`.
- **Model:** [`Tim2190/granite-278m-kk`](https://huggingface.co/Tim2190/granite-278m-kk)
  (fp16, ~556 MB; v2 on `main`, v1 under the `v1` tag).

## Methodological notes

- The primary benchmark's `vocabulary-gap` category, by validation, has high lexical
  overlap with the answer, so it is better read as "discriminating close entities" than
  "understanding synonyms". Clean semantic evidence comes from the OOD paraphrase /
  low_overlap tiers.
- All comparisons use the benchmark's own harness and metrics at seq 512; significance is
  paired bootstrap (10,000 resamples).
