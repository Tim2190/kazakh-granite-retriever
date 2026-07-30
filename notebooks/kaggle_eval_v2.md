# Честный замер v2 на seq 512 (Kaggle)

Замеряем **строго на seq 512** (не 256!) с paired bootstrap. Модель v2 уже в Output
обучающего кернела — не таскаем её, а подключаем как Input к этому кернелу.

Порядок: dense v2 vs zero-shot (база 278m) и vs kazakh-e5 → потом v2 vs v1 →
потом гибрид. **Никаких правок в доках/HF, пока не увидим значимость.**

---

## Подготовка

1. Новый Kaggle Notebook, **Accelerator: GPU T4** (для быстрого энкодинга).
2. **Add Input → Notebook Output →** выбери свой обучающий кернел («Model V2»).
   Модель окажется в `/kaggle/input/<имя-кернела>/granite-278m-kk-v2`.
   Поправь `MODEL_V2` в Ячейке 2 под реальный путь (проверь `!ls /kaggle/input`).

## Ячейка 1 — клоны и зависимости

```python
!pip -q install -U sentence-transformers datasets
!git clone --depth 1 https://github.com/Tim2190/kazakh-granite-retriever.git
!git clone --depth 1 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git bench
%cd kazakh-granite-retriever
!ls /kaggle/input          # найди тут папку granite-278m-kk-v2
```

## Ячейка 2 — dense v2 vs zero-shot (278m) и vs kazakh-e5, seq 512

```python
import os, glob
os.environ["KAZAKH_STEMMER_KEY"] = ""   # понадобится только для --hybrid (Ячейка 4)

# автопоиск папки модели в подключённом Input (slug у Kaggle произвольный)
cands = glob.glob("/kaggle/input/**/granite-278m-kk-v2", recursive=True)
assert cands, "нет granite-278m-kk-v2 в /kaggle/input — Add Input → Notebook Output → кернел Model V2"
MODEL_V2 = cands[0]
print("MODEL_V2 =", MODEL_V2)

!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned {MODEL_V2} \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model shyngys-e5 \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_512.json
```

Смотрим в выводе:
- **v2 (dense) vs zero-shot 278m** — Δ и p по ALL и срезам (должно быть значимо вверх);
- **v2 vs kazakh-e5** (`vs_rows`) — Δ и p; особый интерес к `inflected` (там kazakh-e5
  был значимо сильнее у v1 — 0.836 vs 0.752; негативы били ровно в это).

## Ячейка 3 — v2 vs v1 (прямое сравнение прогресса)

```python
!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned {MODEL_V2} \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model Tim2190/granite-278m-kk \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_vs_v1_512.json
```

`vs_rows` здесь = **v2 против v1** (наша прошлая модель). Хотим ALL значимо вверх.

## Ячейка 4 (опц.) — гибрид с BM25(казахский стеммер), seq 512

```python
import os
os.environ["KAZAKH_STEMMER_KEY"] = ""   # ← вставь ключ стеммера

!python scripts/eval.py \
    --benchmark-root ../bench \
    --finetuned {MODEL_V2} \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --vs-model shyngys-e5 \
    --hybrid --bm25-stemmer kazakh-prod --rrf-k 60 \
    --max-seq-len 512 --top-k 10 --n-resamples 10000 \
    --out results/eval_v2_hybrid_512.json
```

## Ячейка 5 — OOD (речи Akorda), проверка обобщения

```python
!git clone --depth 1 https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ.git ood
!python scripts/eval_ood.py \
    --benchmark-root ../bench \
    --ood-root ../ood \
    --finetuned {MODEL_V2} \
    --base-model ibm-granite/granite-embedding-278m-multilingual \
    --max-seq-len 512 --n-resamples 10000 \
    --out results/eval_v2_ood_512.json
```

---

## После замера

Скинь мне `results/eval_v2_*.json` (или ключевые строки вывода). Дальше — **только по фактам**:

- **v2 значимо лучше v1 и/или обходит/догоняет kazakh-e5 (p<0.05)** → тегаем v1 на HF,
  пушим v2 в `Tim2190/granite-278m-kk`, обновляю README/REPORT честными числами @512.
- **разница в пределах шума** → так и пишем; v2 остаётся как есть, доки не переписываем
  под «стало лучше». Как договаривались после истории с overclaim — сначала значимость.
