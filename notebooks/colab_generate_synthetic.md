# Synthetic generation on Colab

Generates synthetic (query → passage) pairs on a Colab runtime, which persists on its
own regardless of any local session.

The generator is **resumable**: it re-reads `data/synthetic_pairs.jsonl`, skips already
processed `passage_id`s (deterministic `seed=13`), and appends only new pairs — so it can
be stopped and continued without duplicates.

---

## Cell 1 — install and clone

```python
!pip -q install requests

# KazQAD corpus (Kazakh Wikipedia, CC BY-SA) — source of passages
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git

# project repo: script + current synthetic_pairs.jsonl (for resume) + exclude-ids.
# set GH_TOKEN to a GitHub PAT to push results back (repo/public_repo scope).
GH_TOKEN = ""
GH_USER  = "Tim2190"
REPO     = "kazakh-granite-retriever"
BRANCH   = "main"

auth = f"{GH_TOKEN}@" if GH_TOKEN else ""
!git clone --branch {BRANCH} https://{auth}github.com/{GH_USER}/{REPO}.git
%cd {REPO}
!git config user.email "you@example.com" && git config user.name "your-name"
!wc -l data/synthetic_pairs.jsonl
```

## Cell 2 — generate (background, auto-commit progress)

```python
import subprocess, time, os, glob

GEMINI_API_KEY = ""       # Gemini API key
TARGET         = 60000    # target number of pairs
CORPUS_GLOB    = "../KazQAD/data/information-retrieval/corpus/*.jsonl.gz"
BRANCH         = "main"

assert glob.glob(CORPUS_GLOB), "KazQAD corpus not found — check the path"

env = dict(os.environ, GEMINI_API_KEY=GEMINI_API_KEY)
proc = subprocess.Popen(
    ["python3", "scripts/generate_synthetic.py",
     "--corpus", CORPUS_GLOB,
     "--provider", "gemini", "--model", "gemini-3.5-flash-lite",
     "--n-passages", "8000", "--batch", "12", "--sleep", "4.2",
     "--exclude-article-ids", "data/exclude_article_ids.txt",
     "--out", "data/synthetic_pairs.jsonl"],
    env=env,
)

def count():
    return sum(1 for _ in open("data/synthetic_pairs.jsonl", encoding="utf-8"))

# commit progress every 5 minutes until TARGET is reached or the process exits
while True:
    time.sleep(300)
    n = count()
    print("pairs:", n, flush=True)
    os.system('git add data/synthetic_pairs.jsonl && '
              f'git commit -q -m "data: synthetic pairs {n}" && '
              f'git push origin {BRANCH}')
    if n >= TARGET or proc.poll() is not None:
        break

proc.terminate()
print("done, pairs:", count())
```

## Notes

- **Gemini quota.** The free tier has a daily request cap. On `429` /
  `RESOURCE_EXHAUSTED`, re-run Cell 2 the next day — resume continues from where it stopped.
- **PAT.** The token needs `repo` (private push) or `public_repo` scope; it lives only in
  the Colab cell, never committed.
- **Next.** Once the file reaches the target, mine hard negatives
  (`notebooks/colab_mine_negatives.md`), then train (`notebooks/kaggle_train_v2.md`).
