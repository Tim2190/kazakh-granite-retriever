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
(R1). Kazakh is **not** in Granite's official language list — this model adapts it to Kazakh
retrieval with targeted fine-tuning, while staying lightweight (~556 MB, fp16).

- **768-dim** embeddings, cosine similarity, max sequence length **512**, **no prompt prefixes**.
- Current revision is **v2** (morphology-hardened). The previous model is kept as
  `revision="v1"`.
- Full method, data, evaluation and scripts:
  **[github.com/Tim2190/kazakh-granite-retriever](https://github.com/Tim2190/kazakh-granite-retriever)**.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Tim2190/granite-278m-kk")   # ~556 MB, fp16 (v2)
query = "Балқаш көлі қайда орналасқан?"
passages = ["Балқаш — Қазақстанның оңтүстік-шығысындағы тұйық көл ..."]
scores = model.similarity(model.encode([query]), model.encode(passages))
# previous version: SentenceTransformer("Tim2190/granite-278m-kk", revision="v1")
```

No special query/passage prefixes are needed. For stronger retrieval you can fuse the dense
scores with a lexical channel (BM25 + a Kazakh stemmer) via Reciprocal Rank Fusion — see the
[eval scripts](https://github.com/Tim2190/kazakh-granite-retriever) (`eval.py --hybrid`).

## Training

- **Data:** 57,369 synthetic Kazakh (query → passage) pairs over Kazakh Wikipedia passages
  from [KazQAD](https://github.com/IS2AI/KazQAD), each with a **hard negative mined via BM25
  with a Kazakh stemmer**, plus 3,733 KazQAD labeled gold pairs (with their rel=0 negatives)
  — 61,102 training examples. Training data is de-duplicated against the evaluation
  benchmarks (anti-leak).
- **Objective:** CachedMultipleNegativesRankingLoss (sentence-transformers), 2 epochs,
  lr 1e-5, **max_seq_len 512**, Kaggle T4.
- **Base:** `ibm-granite/granite-embedding-278m-multilingual` (Apache-2.0).

## Evaluation

Evaluated at **seq 512** with the benchmark's own harness and paired-bootstrap significance
(10k). Fine-tuning **significantly improves the base Granite** on Kazakh retrieval
(in-domain nDCG@10 0.672 → 0.751 dense / 0.813 hybrid, p<0.001; out-of-domain speeches
0.430 → 0.529, all tiers p<0.05). The model is **on par with the specialized kazakh-e5**
(ALL: statistical tie, p=0.42), and the stemmer-mined hard negatives **significantly
improve morphological (inflected) queries over the previous v1** (p=0.002), closing the
one slice where kazakh-e5 previously led. It does **not** beat kazakh-e5 overall. Full
tables, baselines and significance tests are in the
[project repository](https://github.com/Tim2190/kazakh-granite-retriever).

## License & attribution

Training passages come from **KazQAD** / Kazakh Wikipedia (**CC BY-SA 4.0**); the model is
released under **CC BY-SA 4.0** with attribution to KazQAD and Kazakh Wikipedia. The base
Granite model is Apache-2.0.

## Citation

Please cite the project repository and the underlying benchmarks/datasets
([Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark),
[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ),
[KazQAD](https://github.com/IS2AI/KazQAD)).
