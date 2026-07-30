# Training on Kaggle T4 (synthetic + stemmer hard-negatives + KazQAD gold)

Trains the model on 57,369 synthetic pairs with stemmer-mined hard negatives plus
3,733 KazQAD gold pairs, at **seq 512** (so the model natively handles the 512-token
evaluation).

The full-text training file (~240 MB) is not stored in git — it is assembled on the
machine from the compact artifacts (`data/synthetic_pairs.hn.ids.jsonl`,
`data/kazqad_pairs.dedup.jsonl`) and the KazQAD corpus.

Run as **Save & Run All (Commit)** for an unattended 12-hour session; the model is
written to Output.

## Cell 1 — dependencies, clones, build the train file

```python
!pip -q install -U sentence-transformers datasets accelerate

!git clone --depth 1 https://github.com/Tim2190/kazakh-granite-retriever.git
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git
%cd kazakh-granite-retriever

!python scripts/build_train_v2.py \
    --ids  data/synthetic_pairs.hn.ids.jsonl \
    --gold data/kazqad_pairs.dedup.jsonl \
    --corpus '../KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \
    --out data/train_pairs_v2.jsonl
```

## Cell 2 — training

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!python scripts/train.py \
    --data data/train_pairs_v2.jsonl \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --output-dir /kaggle/working/granite-278m-kk-v2 \
    --epochs 2 --batch-size 128 --mini-batch 8 \
    --lr 1e-5 --max-seq-len 512 \
    --max-neg-per-pair 1 --fp16
```

- **`--fp16` is required on T4.** Without it training runs in fp32 (~57 s/step, ~15 h for
  2 epochs on a single T4 — over the 12-hour session limit). With `--fp16` and 2×T4 the
  effective batch is 256, ~476 steps total, ~7.5 h. The model stays in fp32 as master
  weights; fp16 is autocast only.
- **`--mini-batch 8`** fits T4 16 GB at seq 512; drop to `4` on `CUDA out of memory`
  (CachedMNRL accumulates, so the effective batch is unchanged).
- **`--max-neg-per-pair 1`** — one mined hard negative per pair (+127 in-batch), ~60K
  triplets. Raise to `2` for more, at the cost of a longer run.
- The model is saved fp16 (~556 MB) to `/kaggle/working/granite-278m-kk-v2`.

## Cell 3 (optional) — publish to the Hub

Publishes the trained model. The guard skips the push when no token is set, so
`Save & Run All` does not fail on an empty token.

```python
import os
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    # keep the current Hub revision as a tag before overwriting:
    HfApi(token=os.environ["HF_TOKEN"]).create_tag(
        "Tim2190/granite-278m-kk", tag="v1", revision="main")
    m = SentenceTransformer("/kaggle/working/granite-278m-kk-v2")
    m.push_to_hub("Tim2190/granite-278m-kk", token=os.environ["HF_TOKEN"], exist_ok=True)
else:
    print("HF_TOKEN not set — skipping push.")
```

Evaluation is a separate notebook: `notebooks/kaggle_eval_v2.md`.
