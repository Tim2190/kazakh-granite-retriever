# Переобучение v2 на Kaggle T4 (57K синтетика + стеммер-негативы + KazQAD gold)

Отличие от v1: (1) 57K синтетики вместо 40K, (2) **hard-negatives, намайненные
казахским стеммером** (по 1–2 на пару), (3) обучение на **seq 512** — чтобы модель
училась на полных пассажах и нативно работала на честном замере @512 (у v1 был
разрыв: обучали на 256, мерили на 512).

Тяжёлый train-файл (~240 МБ) в репо не лежит — собирается на месте из лёгких
артефактов (`hn.ids.jsonl` 16 МБ + `kazqad_pairs.dedup.jsonl` 7 МБ + корпус KazQAD).

Запусти как **Save & Run All (Commit)** — тогда 12-часовая сессия отработает без
тебя, а модель осядет в Output.

---

## Ячейка 1 — зависимости, клоны, сборка train-файла

```python
!pip -q install -U sentence-transformers datasets accelerate

!git clone --depth 1 https://github.com/Tim2190/kazakh-granite-retriever.git
!git clone --depth 1 https://github.com/IS2AI/KazQAD.git
%cd kazakh-granite-retriever

# собрать полнотекстовый train-файл (id → тексты из корпуса) + KazQAD gold
!python scripts/build_train_v2.py \
    --ids  data/synthetic_pairs.hn.ids.jsonl \
    --gold data/kazqad_pairs.dedup.jsonl \
    --corpus '../KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \
    --out data/train_pairs_v2.jsonl
!wc -l data/train_pairs_v2.jsonl
```

## Ячейка 2 — обучение

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = ""     # только если будешь пушить сразу на HF (иначе пусто)

!python scripts/train.py \
    --data data/train_pairs_v2.jsonl \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --output-dir /kaggle/working/granite-278m-kk-v2 \
    --epochs 2 --batch-size 128 --mini-batch 8 \
    --lr 1e-5 --max-seq-len 512 \
    --max-neg-per-pair 1
```

- **`--max-neg-per-pair 1`** — 1 сильный намайненный негатив на пару (+127 in-batch).
  Даёт ~60K триплетов, влезает в 12-часовую сессию на seq 512 с запасом.
  Есть время/GPU-квота — можно поднять до `2` (≈115K триплетов, дольше).
- **`--mini-batch 8`** — под T4 16 ГБ на seq 512. Если `CUDA out of memory` —
  поставь `4` (на эффективный батч 128 это не влияет, CachedMNRL аккумулирует).
- Сохранение fp16 по умолчанию (~556 МБ). Модель → `/kaggle/working/granite-278m-kk-v2`.

## Ячейка 3 (опц.) — сразу запушить на HF как ревизию v2

Только если замер уже подтвердит, что v2 значимо лучше. Иначе сначала eval@512.

```python
# сохранить v1 как ревизию ПЕРЕД перезаписью (один раз, из среды с доступом к HF):
#   huggingface-cli repo tag create Tim2190/granite-278m-kk v1
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("/kaggle/working/granite-278m-kk-v2")
m.push_to_hub("Tim2190/granite-278m-kk", token=os.environ["HF_TOKEN"])
```

## Дальше — честный замер (не пропускать!)

Скачай `granite-278m-kk-v2` из Output и прогони `eval.py` **на seq 512** с
`--vs-model shyngys-e5` и paired bootstrap (см. `notebooks/colab_eval_v2.md`).
Только после значимого выигрыша на 512 — перезапись `main` на HF и правки в доках.
