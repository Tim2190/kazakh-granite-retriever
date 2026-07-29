---
language:
- kk
license: cc-by-sa-4.0
base_model: ibm-granite/granite-embedding-278m-multilingual
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- kazakh
- retrieval
- feature-extraction
- granite
---

# Granite-278m-kk — Kazakh retrieval embedding

A compact (278M) sentence-embedding model for **Kazakh** search and RAG, fine-tuned from
[`ibm-granite/granite-embedding-278m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)
(R1). Kazakh is **not** in Granite's official language list — this model shows that
targeted fine-tuning closes that gap in practice: it outperforms the specialized Kazakh
fine-tunes, improves the base on two independent domains, and stays lightweight (~556 MB, fp16).

- **768-dim** embeddings, cosine similarity, max sequence length **256**, **no prompt prefixes**.
- Full method, data and scripts: **[github.com/Tim2190/kazakh-granite-retriever](https://github.com/Tim2190/kazakh-granite-retriever)**.

## Results (nDCG@10)

**Primary benchmark** — [Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
(Kazakh Wikipedia, 300 queries, 8,370 passages):

| # | system | ALL |
|---|---|---|
| 1 | bge-m3 | 0.866 |
| 2 | jina-v3 | 0.821 |
| **3** | **Granite-278m-kk ⊕ BM25 (hybrid)** | **0.814** |
| 4 | kazakh-e5 ⊕ BM25 | 0.808 |
| 6 | multilingual-e5-base | 0.785 |
| **8** | **Granite-278m-kk (this model, dense)** | **0.752** |
| 9 | kazakh-e5 (specialized) | 0.747 |
| 10 | Granite-R1 278m (base, zero-shot) | 0.672 |

The dense model beats the specialized Kazakh **kazakh-e5** (0.752 vs 0.747) and the base
Granite (0.672 → 0.752); the BM25 hybrid beats the kazakh-e5⊕BM25 reference (0.808) and
e5-base (0.785). Strong general models (bge-m3, jina-v3) remain ahead.

**Independent OOD check** — [RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)
(official speeches, a different domain, unseen in training): the model improves over the
base **on every tier significantly** (ALL 0.428 → 0.507; factoid, paraphrase, low_overlap
all p<0.05) — the gain generalizes, it is not overfitting to one benchmark.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Tim2190/granite-278m-kk")   # ~556 MB, fp16
query = "Балқаш көлі қайда орналасқан?"
passages = ["Балқаш — Қазақстанның оңтүстік-шығысындағы тұйық көл ..."]
scores = model.similarity(model.encode([query]), model.encode(passages))
```

No special query/passage prefixes are needed. For the strongest setup, fuse the dense
scores with BM25 (+ a Kazakh stemmer) via Reciprocal Rank Fusion — see the
[eval scripts](https://github.com/Tim2190/kazakh-granite-retriever) (`eval.py --hybrid`).

## Training

- **Data:** 40,084 synthetic Kazakh (query → passage) pairs generated over Kazakh
  Wikipedia passages from [KazQAD](https://github.com/IS2AI/KazQAD), plus KazQAD's
  labeled gold pairs and hard negatives. Training data is de-duplicated against the
  evaluation benchmarks (anti-leak).
- **Objective:** CachedMultipleNegativesRankingLoss (sentence-transformers), 2 epochs,
  lr 1e-5, max_seq_len 256, single T4 (~1.9 h).
- **Base:** `ibm-granite/granite-embedding-278m-multilingual` (Apache-2.0).

## License & attribution

Training passages come from **KazQAD** / Kazakh Wikipedia (**CC BY-SA 4.0**); the model is
released under **CC BY-SA 4.0** with attribution to KazQAD and Kazakh Wikipedia. The base
Granite model is Apache-2.0.

## Citation

Please cite the project repository and the underlying benchmarks/datasets
(Kaz-RAG-search-benchmark, RAG-Two-Pass-Retrieval-QAZ, KazQAD) linked above.
