# Evaluation at seq 512 (Kaggle)

Evaluates the model at **seq 512** with paired bootstrap (10k), reusing the benchmark's
own harness and metrics. The trained model is taken from a Kaggle notebook's Output and
attached as an Input, so nothing needs to be downloaded or moved.

Runs: dense vs zero-shot base, dense vs kazakh-e5, dense vs the previous revision, the
BM25 hybrid, and the OOD (speeches) set.

## Setup

1. New Kaggle Notebook, **Accelerator: GPU T4**.
2. **Add Input → Notebook Output →** the training notebook, so the model appears under
   `/kaggle/input/<slug>/granite-278m-kk-v2`. The cells below auto-discover the path.

## Cell 1 — clones and dependencies

```python
!pip -q install -U sentence-transformers datasets
!git clone --depth 1 https://github.com/Tim2190/kazakh-granite-retriever.git
!git clone --depth 1 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git bench
%cd kazakh-granite-retriever
```

## Cell 2 — dense vs zero-shot base and vs kazakh-e5, seq 512

```python
import glob
cands = glob.glob("/kaggle/input/**/granite-278m-kk-v2", recursive=True)
assert cands, "model not found in /kaggle/input — Add Input → Notebook Output → training notebook"
MODEL = cands[0]

!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned "{MODEL}" \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model shyngys-e5 \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_512.json
```

Output reports the fine-tuned vs zero-shot table (per slice, with p) and a `vs_rows`
block for the paired comparison against kazakh-e5.

## Cell 3 — vs the previous revision

```python
MODEL = glob.glob("/kaggle/input/**/granite-278m-kk-v2", recursive=True)[0]
!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned "{MODEL}" \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model Tim2190/granite-278m-kk \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_vs_prev_512.json
```

## Cell 4 — hybrid with BM25 (Kazakh stemmer), seq 512

```python
import os
os.environ["KAZAKH_STEMMER_KEY"] = ""   # X-API-Key of the Kazakh stemmer service
MODEL = glob.glob("/kaggle/input/**/granite-278m-kk-v2", recursive=True)[0]
!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned "{MODEL}" \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model shyngys-e5 \
    --hybrid --bm25-stemmer kazakh-prod --rrf-k 60 \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_hybrid_512.json
```

## Cell 5 — OOD (Akorda speeches), generalization check

```python
!git clone --depth 1 https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ.git ood
MODEL = glob.glob("/kaggle/input/**/granite-278m-kk-v2", recursive=True)[0]
!python scripts/eval_ood.py \
    --benchmark-root ../bench \
    --ood-root ood \
    --finetuned "{MODEL}" \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --max-seq-len 512 --n-resamples 10000 \
    --out results/eval_v2_ood_512.json
```

Notes:
- All runs use `--max-seq-len 512`; embeddings are computed fresh in a clean kernel.
- `eval_ood.py` compares the fine-tuned model against zero-shot only (no `--vs-model`).
- The stemmer endpoint (`run.app`) and HF model downloads require open internet, available
  on Kaggle.
