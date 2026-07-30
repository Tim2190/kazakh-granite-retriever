# Results — Granite-278m-kk v2, seq 512

Benchmark: [Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
(Kazakh Wikipedia, 300 queries, 8,370 passages) and the OOD
[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)
(Akorda speeches). Same harness and metrics as the benchmarks; significance is paired
bootstrap (10,000 resamples). All numbers at **max_seq_len 512**.

## Primary benchmark (Wikipedia), nDCG@10

**Zero-shot vs v2 (dense):**

| slice | zero-shot | v2 | Δ | p |
|---|---|---|---|---|
| ALL | 0.672 | 0.751 | +0.079 | <0.001 |
| natural | 0.923 | 0.928 | +0.004 | 0.384 |
| inflected | 0.791 | 0.792 | +0.002 | 0.475 |
| vocabulary-gap | 0.303 | 0.534 | +0.231 | <0.001 |

**Hybrid (dense ⊕ BM25 with Kazakh stemmer, RRF):**

| slice | dense | BM25(kaz) | hybrid |
|---|---|---|---|
| ALL | 0.751 | 0.757 | 0.813 |
| natural | 0.928 | 0.772 | 0.888 |
| inflected | 0.792 | 0.736 | 0.822 |
| vocabulary-gap | 0.534 | 0.764 | 0.728 |

**v2 vs kazakh-e5 (paired bootstrap):**

| slice | v2 | kazakh-e5 | Δ | p |
|---|---|---|---|---|
| ALL | 0.751 | 0.747 | +0.004 | 0.419 (tie) |
| inflected | 0.792 | 0.836 | −0.044 | 0.063 (tie) |
| natural | 0.928 | 0.909 | +0.019 | 0.213 |
| vocabulary-gap | 0.534 | 0.497 | +0.037 | 0.168 |

On ALL this is a statistical tie. For v1, kazakh-e5 led on inflected significantly
(p=0.001); for v2 that gap is no longer significant (p=0.063).

**v2 vs v1 (effect of the stemmer hard-negatives):**

| slice | v2 | v1 | Δ | p |
|---|---|---|---|---|
| ALL | 0.751 | 0.742 | +0.009 | 0.084 |
| inflected | 0.792 | 0.752 | +0.040 | 0.002 |
| natural | 0.928 | 0.921 | +0.007 | 0.106 |
| vocabulary-gap | 0.534 | 0.553 | −0.019 | 0.055 |

Only inflected improves significantly (p=0.002) — the stemmer negatives' target. The ALL
gain is not significant (p=0.084); vocab-gap dips slightly (p=0.055, borderline).

## OOD (Akorda speeches), nDCG@10

| tier | zero-shot | v2 | Δ | p |
|---|---|---|---|---|
| ALL | 0.430 | 0.529 | +0.099 | <0.001 |
| factoid | 0.548 | 0.680 | +0.132 | <0.001 |
| paraphrase | 0.406 | 0.503 | +0.097 | 0.004 |
| low_overlap | 0.339 | 0.405 | +0.066 | 0.009 |

Hybrid on Akorda: 0.554. The gain generalizes to a different domain, including the
genuinely semantic tiers (paraphrase, low_overlap).

## Summary

v2 significantly improves the base Granite in-domain and out-of-domain, is a statistical
tie with kazakh-e5 on ALL, and is significantly stronger than v1 on morphology (inflected),
closing the one slice where kazakh-e5 previously led. It does not beat kazakh-e5 overall.
