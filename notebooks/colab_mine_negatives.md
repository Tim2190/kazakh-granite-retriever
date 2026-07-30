# Hard-negative mining with the Kazakh stemmer (Colab)

The stemmer endpoint is a Cloud Run service (`run.app`), so mining runs where that host is
reachable (e.g. Colab). The client has retries + backoff, splits a batch on failure,
distinguishes 401/429/5xx, and honors a `--max-stem-requests` budget guard. The stemming
cache (`data/stem_cache.json`) is resumable — re-run the cell after any interruption.

Budget estimate: a pool of ~30K passages needs ~200–250 stemmer requests (well under the
free-tier monthly cap).

---

## Cell 1 — clones and dependencies

```python
!pip -q install numpy

GH_TOKEN = ""            # GitHub PAT (repo scope) for clone and for pushing results
GH_USER  = "Tim2190"
BRANCH   = "main"
auth = f"{GH_TOKEN}@" if GH_TOKEN else ""

# project repo (script, synthetic_pairs.jsonl, exclude-ids, stem cache if present)
!git clone --branch {BRANCH} https://{auth}github.com/{GH_USER}/kazakh-granite-retriever.git
# benchmark — provides the tokenizer src.preprocess.tokenize
!git clone --depth 1 https://{auth}github.com/{GH_USER}/Kaz-RAG-search-benchmark.git bench
# KazQAD corpus (negative candidates)
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git

%cd kazakh-granite-retriever
!git config user.email "you@example.com" && git config user.name "your-name"
```

## Cell 2 — mining

```python
import os
os.environ["KAZAKH_STEMMER_KEY"] = ""     # X-API-Key of the stemmer service

!python scripts/mine_hard_negatives.py \
    --benchmark-root ../bench \
    --pairs data/synthetic_pairs.jsonl \
    --corpus '../KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \
    --exclude-article-ids data/exclude_article_ids.txt \
    --stemmer kazakh-prod \
    --pool-sample 30000 --n-neg 4 --neg-rank-start 2 --top-k 50 \
    --max-stem-requests 900 \
    --stem-cache data/stem_cache.json \
    --out data/synthetic_pairs.hn.jsonl
```

## Cell 3 — push a compact ids-only artifact back

The full-text output is large (hundreds of MB). Only the query + passage ids are pushed;
`build_train_v2.py` re-expands the texts from the corpus at training time.

```python
import json
src, dst = "data/synthetic_pairs.hn.jsonl", "data/synthetic_pairs.hn.ids.jsonl"
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
    for line in f:
        d = json.loads(line)
        o.write(json.dumps({
            "query": d["query"],
            "positive_id": d.get("positive_id") or d.get("passage_id", ""),
            "negative_ids": d.get("negative_ids", []),
            "category": d.get("category"),
        }, ensure_ascii=False) + "\n")

!git add -f data/synthetic_pairs.hn.ids.jsonl
!git commit -m "hard-negatives (ids-only): kazakh-prod stemmer, pool 30k"
!git push origin main
```

## Notes

- **Interrupted mid-mining?** Re-run Cell 2 — `data/stem_cache.json` is picked up and no
  stemmer requests are repeated.
- **`--neg-rank-start 2`** takes negatives from BM25 rank ≥2 (not top-1, which may be
  incidentally relevant); the positive itself and same-article passages are always excluded
  (false-negative guard).
- **Next:** `build_train_v2.py` assembles the training file from the ids artifact; see
  `notebooks/kaggle_train_v2.md`.
