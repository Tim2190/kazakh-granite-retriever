# Майнинг hard-negatives казахским стеммером (Colab)

Эндпоинт стеммера — Cloud Run (`run.app`), из основной среды он заблокирован,
поэтому майнинг гоним на Colab (там сеть до стеммера есть).

Клиент упрочнён: ретраи+backoff, дробление батча при сбое, различение
401/429/5xx, budget-guard `--max-stem-requests`. Кэш стемминга (`data/stem_cache.json`)
резюмируемый — при обрыве просто перезапусти ячейку.

Бюджет (оценка на наших данных): pool≈30K → **~200–250 запросов** к стеммеру
(Free-план — 1000/мес), с запасом.

---

## Ячейка 1 — клоны и зависимости

```python
!pip -q install numpy

GH_TOKEN = ""            # github PAT (repo) — для clone и push результата
GH_USER  = "Tim2190"
BRANCH   = "main"
auth = f"{GH_TOKEN}@" if GH_TOKEN else ""

# наш репо (скрипт, synthetic_pairs.jsonl, exclude-ids, кэш стеммера если уже есть)
!git clone --branch {BRANCH} https://{auth}github.com/{GH_USER}/kazakh-granite-retriever.git
# бенчмарк — оттуда берётся токенайзер src.preprocess.tokenize
!git clone --depth 1 https://{auth}github.com/{GH_USER}/Kaz-RAG-search-benchmark.git bench
# корпус KazQAD (кандидаты негативов)
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git

%cd kazakh-granite-retriever
!git config user.email "9189920ts@gmail.com" && git config user.name "Tim2190"
```

## Ячейка 2 — майнинг

```python
import os
os.environ["KAZAKH_STEMMER_KEY"] = ""     # X-API-Key стеммера

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

## Ячейка 3 — вернуть результат в репо

```python
# кэш стемминга коммитим тоже — резюмируемость + не тратить запросы повторно
!wc -l data/synthetic_pairs.hn.jsonl
!git add data/synthetic_pairs.hn.jsonl data/stem_cache.json
!git commit -m "hard-negatives: kazakh-prod stemmer, pool 30k (colab)"
!git push origin {BRANCH}
```

## Заметки

- **Прервалось на стеммере?** Перезапусти Ячейку 2 — кэш `data/stem_cache.json`
  подхватится, повторных запросов к сервису не будет.
- **`--neg-rank-start 2`** — берём негативы не с топ-1 BM25 (там может случайно
  оказаться релевантное), а с ранга ≥2; плюс из негативов жёстко исключаются сам
  positive и пассажи той же статьи (защита от false-negative).
- **Дальше:** `data/synthetic_pairs.hn.jsonl` идёт в `train.py` — негативы там
  разворачиваются в триплеты `(query, positive, hard_neg)` (`--max-neg-per-pair`).
