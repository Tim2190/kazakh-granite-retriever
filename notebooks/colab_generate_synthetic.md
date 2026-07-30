# Генерация синтетики на Colab (переживает ночь)

Локальная/эфемерная среда убивает фоновый процесс, как только сессия засыпает.
Colab-runtime живёт часами сам по себе — гоняем добор синтетики здесь.

Скрипт **резюмируемый**: перечитывает `data/synthetic_pairs.jsonl`, пропускает уже
обработанные `passage_id` (детерминированный `seed=13`), дописывает только новое.
Поэтому можно останавливать/продолжать без дублей.

---

## Ячейка 1 — установка и клоны

```python
!pip -q install requests

# Корпус KazQAD (казвики, CC BY-SA) — источник пассажей
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git

# Наш репозиторий: скрипт + текущий synthetic_pairs.jsonl (для резюма) + exclude-ids
# Если репо приватный — вставь PAT в GH_TOKEN ниже; если публичный — можно без него.
GH_TOKEN = ""            # github PAT с правом repo (для clone приватного и для push)
GH_USER  = "Tim2190"
REPO     = "kazakh-granite-retriever"
BRANCH   = "main"

import os
url = f"https://{GH_TOKEN+'@' if GH_TOKEN else ''}github.com/{GH_USER}/{REPO}.git"
!git clone --branch {BRANCH} {url}
%cd {REPO}
!git config user.email "9189920ts@gmail.com"
!git config user.name  "Tim2190"
!wc -l data/synthetic_pairs.jsonl
```

## Ячейка 2 — генерация (в фоне, с автокоммитом каждые ~N минут)

```python
import subprocess, time, os, glob

GEMINI_API_KEY = ""       # вставь ключ Gemini
TARGET         = 60000    # целевое число пар
CORPUS_GLOB    = "../KazQAD/data/information-retrieval/corpus/*.jsonl.gz"

assert glob.glob(CORPUS_GLOB), "корпус KazQAD не найден — проверь путь"

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

# автокоммит прогресса каждые 5 минут, пока не дойдём до TARGET или процесс не завершится
while True:
    time.sleep(300)
    n = count()
    print("пар:", n, flush=True)
    os.system('git add data/synthetic_pairs.jsonl && '
              f'git commit -q -m "data: synthetic pairs {n} (colab)" && '
              f'git push origin {BRANCH}')
    if n >= TARGET or proc.poll() is not None:
        break

proc.terminate()
print("готово, пар:", count())
```

## Заметки

- **Квота Gemini.** Free-tier имеет дневной лимит запросов. Если упрёмся (в логе
  `429`/`RESOURCE_EXHAUSTED`) — просто перезапусти Ячейку 2 на следующий день,
  резюм продолжит с места.
- **PAT.** Токену нужен scope `repo` (push в приватный) или `public_repo`.
  Не коммить токен в код репозитория — он живёт только в ячейке Colab.
- **Что дальше.** Как файл дорастёт до цели — переобучение на 60K
  (см. `scripts/train.py`), затем замер строго на seq 512 с paired bootstrap.
