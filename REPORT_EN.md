# Granite-278m-kk — Kazakh retrieval with a fine-tuned IBM Granite

*Русская версия — [REPORT.md](REPORT.md).*

A compact (278M) embedding model for Kazakh search and RAG, fine-tuned from
`ibm-granite/granite-embedding-278m-multilingual` (R1). Kazakh is **not** in
Granite's official language list — this model shows that targeted fine-tuning
closes that gap in practice: it outperforms the specialized Kazakh models and
markedly improves the base, while staying lightweight and reproducible.

---

## At a glance

- **Beats the specialized Kazakh fine-tunes** (kazakh-e5, kazembed-v5) on two
  independent domains.
- **The BM25 hybrid beats the previous reference top** kazakh-e5 ⊕ BM25 (0.814 vs
  0.808) and base multilingual-e5-base (0.785) on the primary benchmark.
- **Significantly improves the base Granite-R1**: Wiki 0.672 → 0.752 (dense) /
  0.814 (hybrid).
- **The gain generalizes** to an independent OOD domain (official speeches) —
  significant across every tier.
- Compact (278M), trained on a single T4 (~1h45m), fully reproducible.

## Where the model stands (honestly)

Primary benchmark [Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
(Wikipedia, 300 queries, 8,370 passages), nDCG@10 (ALL):

| # | system | ALL |
|---|---|---|
| 1 | bge-m3 | 0.866 |
| 2 | jina-v3 | 0.821 |
| **3** | **Granite-278m-kk ⊕ BM25 (ours)** | **0.814** |
| 4 | kazakh-e5 ⊕ BM25 | 0.808 |
| 5 | cohere embed-v4 | 0.800 |
| 6 | multilingual-e5-base | 0.785 |
| 7 | BM25 + Kazakh stemmer | 0.754 |
| **8** | **Granite-278m-kk (ours, dense)** | **0.752** |
| 9 | kazakh-e5 (specialized) | 0.747 |
| 10 | Granite-R1 278m (base, zero-shot) | 0.672 |

The model sits in the top tier: the dense version beats the specialized Kazakh
kazakh-e5, and the BM25 hybrid beats its hybrid and the base e5. Strong general
multilingual models (bge-m3, jina-v3) remain ahead — a target for further work.

## Benchmarks & sources

- **[Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)** —
  the primary benchmark (Kazakh Wikipedia). Source of the comparative leaderboard and the
  eval harness: `src/retrieval` (DenseIndex, BM25 + Kazakh stemmer), `src/eval` (metrics,
  paired bootstrap). Our `eval.py` / `eval_ood.py` import it directly — numbers are
  computed by the same code as the benchmark's.
- **[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)** —
  the independent OOD benchmark (official speeches, akorda.kz / nazarbayev.kz).
- **[KazQAD](https://github.com/IS2AI/KazQAD)** — training passages (Kazakh Wikipedia,
  CC BY-SA 4.0).

## How it was built

Full pipeline — 8 scripts in `scripts/`.

1. **Base selection (`zeroshot_107m.py`).** Zero-shot of all Granite models.
   **278m R1** (the strongest zero-shot Granite, 0.672) was chosen as the flagship.
2. **Data (`prepare_data.py`).** KazQAD (Kazakh Wikipedia, CC BY-SA): 825K passages
   + labeled triples → 3,893 query→gold pairs + hard negatives (`rel=0`).
3. **Anti-leak (`check_overlap.py`).** Dedup of train against the eval benchmark
   (both derive from Kazakh Wikipedia) by title + near-dup + article_id →
   **3,733 clean pairs**.
4. **Scaling (`generate_synthetic.py`).** Synthetic question generation (Gemini) over
   Kazakh Wikipedia passages, balanced by type, with strict anti-leak.
   Result: **40,084 pairs** across 11,929 distinct articles.
5. **Training (`train.py`).** sentence-transformers, CachedMNRL, 278m, 2 epochs,
   lr 1e-5, max_seq_len 256, fp32, Kaggle T4. Negatives: in-batch (for synthetic) +
   labeled KazQAD negatives (`rel=0`) for gold pairs.
6. **Evaluation (`eval.py`, `eval_ood.py`).** Same harness and metrics as the
   benchmark, paired bootstrap (10k). Hybrid — RRF(dense, BM25 + Kazakh stemmer).

> **What is NOT in the final model.** The Kazakh stemmer is used **only in the
> hybrid's BM25 channel** at evaluation, not during training. Separately, BM25
> hard-negative mining for the training pairs (`mine_hard_negatives.py`, identity
> stemmer) was explored: on 14.7K it did **not** improve ALL and added an inflected
> regression, so it is not in the final recipe. The decisive lever was data volume.

## Results — primary benchmark (Wikipedia)

**Zero-shot vs fine-tuned (nDCG@10):**

| slice | zero-shot | fine-tuned | Δ | p |
|---|---|---|---|---|
| **ALL** | 0.671 | **0.752** | +0.081 | **<0.001** |
| natural | 0.920 | 0.929 | +0.008 | 0.300 |
| inflected | 0.791 | 0.766 | −0.025 | 0.128 |
| vocabulary-gap | 0.301 | 0.562 | +0.260 | <0.001 |

Significant ALL gain, natural preserved (even up), no significant regressions.

**Hybrid with BM25 (Kazakh stemmer):**

| slice | dense | BM25(kaz) | **hybrid** |
|---|---|---|---|
| **ALL** | 0.752 | 0.757 | **0.814** |
| natural | 0.929 | 0.772 | 0.884 |
| inflected | 0.766 | 0.736 | 0.805 |
| vocabulary-gap | 0.562 | 0.764 | 0.752 |

The hybrid (0.814) is above kazakh-e5 ⊕ BM25 (0.808) and e5-base (0.785).

**Key comparison with Kazakh fine-tunes:** kazakh-e5 (0.747) and kazembed-v5 (0.642)
— specialized Kazakh fine-tunes of multilingual-e5 — end up **below** our model
(0.752) and below their own base e5 (0.785). Our fine-tune, by contrast, **improved**
the base. Takeaway: a properly targeted fine-tune beats a "naive" Kazakh one.

## Results — independent validation (OOD: official speeches, Akorda)

Second benchmark [RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ):
471 passages from akorda.kz / nazarbayev.kz speeches — a **different domain**, not
seen in training. A generalization check.

**Zero-shot vs fine-tuned — every tier significantly up:**

| tier | zero-shot | fine-tuned | Δ | p |
|---|---|---|---|---|
| **ALL** | 0.428 | 0.507 | +0.079 | <0.001 |
| factoid | 0.549 | 0.659 | +0.110 | <0.001 |
| paraphrase | 0.406 | 0.474 | +0.068 | 0.019 |
| low_overlap | 0.332 | 0.389 | +0.057 | 0.027 |

The gain transfers to another domain across all query types, including the genuinely
semantic ones (paraphrase, low_overlap) — a real skill, not overfitting to the first
benchmark. On Akorda our model (dense 0.507, hybrid 0.552) beats base Granite-R1
(0.431) and both Kazakh fine-tunes (shyngys/kazakh-e5 0.426, kazembed-v5 0.389).
In the full Akorda field the strong general models (bge-m3 0.679, jina-v3 0.613)
are ahead.

## Future work (further development)

The experiment succeeded: fine-tuning turned an officially "unsupported" Granite into
a competitive Kazakh retriever that beats specialized Kazakh models and improves the
base on two domains. Untested levers that could push further, given motivation:

1. **More synthetic data: 40K → 60K+ pairs.** Volume was the decisive lever
   (14.7K → 40K: 0.714 → 0.752), and the curve has not plateaued; the KazQAD corpus
   (825K passages) scales almost without limit.
2. **Quality hard negatives via the Kazakh stemmer.** The one **untested** version of
   this lever. Identity-BM25 mining gave nothing (weak decoys), but the stemmer finds
   real morphological traps. The miner is ready
   (`mine_hard_negatives.py --stemmer kazakh-prod`) — run on the full set and retrain.
   **60K + stemmer negatives** is the most obvious next step.
3. **Hybrid as a single component.** Wrap Granite-ft + BM25(Kazakh stemmer) + RRF into
   one `.search()` retriever — a production-ready artifact.
4. **Training tuning.** More epochs/data on 278m, Matryoshka representations, LR / MNRL
   temperature search.
5. **Analyze the bge-m3 / jina-v3 gap** and port applicable practices into the Kazakh
   Granite fine-tune.

## Usage

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("<HF model repo>")            # link to be added
emb = m.encode(["Балқаш көлі қайда орналасқан?"])     # no special prefixes
```

For the strongest setup, use the BM25 hybrid (RRF); see `eval.py --hybrid` and `scripts/`.

## Data, license, reproducibility

- **Training data:** `data/synthetic_pairs.jsonl` (40,084 pairs) — synthetic over
  KazQAD passages (Kazakh Wikipedia, **CC BY-SA 4.0**, attribution to KazQAD).
- **Scripts:** `zeroshot_107m.py`, `prepare_data.py`, `check_overlap.py`,
  `generate_synthetic.py`, `mine_hard_negatives.py`, `train.py`, `eval.py`, `eval_ood.py`.
- **Run reports:** `results/*.md`.

## Methodological notes

- The primary benchmark's `vocabulary-gap` category, by validation, has high lexical
  overlap with the answer (descriptive questions with rare keywords), so it is better
  read as "discriminating close entities" than "understanding synonyms". Clean semantic
  evidence comes from the OOD paraphrase / low_overlap tiers.
- All comparisons use the benchmark's own harness and metrics; significance is paired
  bootstrap (10,000 resamples).
