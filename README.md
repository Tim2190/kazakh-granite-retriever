# kazakh-granite-retriever

*English — [README_EN.md](README_EN.md).*

**🤗 Модель на HuggingFace: [`Tim2190/granite-278m-kk`](https://huggingface.co/Tim2190/granite-278m-kk)** (fp16, ~556 МБ).
Текущая ревизия — **v2** (морфологически усиленная); предыдущая доступна как `revision="v1"`.

**Granite-278m-kk** — компактная (278M) embedding-модель для казахского поиска и RAG,
дообученная из `ibm-granite/granite-embedding-278m-multilingual` (R1). Казахского в
официальном списке Granite нет — модель показывает, что таргетированный файнтюн
закрывает это на практике.

> **📄 Полный отчёт — [REPORT.md](REPORT.md)** (метод, все таблицы, значимость).

## Итог (честно, по значимости)

Все сравнения — на seq 512, тем же харнессом и метриками, что и у бенчмарка,
значимость — paired bootstrap (10k).

- **Значимо улучшает базовый Granite-R1** и на основном домене (Wiki: 0.672 → **0.751**
  dense / **0.813** hybrid, p<0.001), и на независимом OOD (речи: 0.430 → **0.529**,
  все срезы p<0.05).
- **Идёт вровень со специализированным kazakh-e5** — по ALL статистическая ничья
  (0.751 vs 0.747, p=0.42). Обгона нет, и утверждать его мы не будем.
- **v2 значимо крепче v1 на морфологии** (inflected 0.752 → 0.792, p=0.002) — эффект
  hard-negatives, намайненных казахским стеммером. Тем самым **закрыто единственное
  значимое преимущество kazakh-e5** (морфология: было p=0.001 в его пользу → стало
  p=0.06, ничья).

**Основной бенчмарк (Википедия), nDCG@10 @512:**

| срез | zero-shot | v2 dense | v2 ⊕ BM25 | v2 vs kazakh-e5 (Δ, p) |
|---|---|---|---|---|
| **ALL** | 0.672 | **0.751** | **0.813** | +0.004, p=0.42 (ничья) |
| inflected | 0.791 | 0.792 | 0.822 | −0.044, p=0.06 (ничья) |
| natural | 0.923 | 0.928 | 0.888 | +0.019, p=0.21 |
| vocabulary-gap | 0.303 | 0.534 | 0.728 | +0.037, p=0.17 |

Где стоит модель (парного теста между чужими системами у нас нет, поэтому —
«примерно»): гибрид **0.813** — в одном ряду с kazakh-e5 ⊕ BM25 (~0.808) и выше
e5-base (~0.785); сильные общие мультиязычные модели (bge-m3 ~0.866, jina-v3 ~0.821)
впереди — ориентир для развития.

## Бенчмарки и источники

Рейтинги и оценка — на этих бенчмарках; наши `eval.py` / `eval_ood.py` **переиспользуют
их харнесс и метрики** для сопоставимости чисел:

- **[Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)** —
  основной бенчмарк (казахская Википедия, 300 запросов, 8 370 пассажей). Источник
  сравнительного рейтинга и eval-харнесса: `src/retrieval` (DenseIndex, BM25 + казахский
  стеммер), `src/eval` (метрики, paired bootstrap) — импортируются нашими скриптами напрямую.
- **[RAG-Two-Pass-Retrieval-QAZ](https://github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ)** —
  независимый OOD-бенчмарк (официальные речи akorda.kz / nazarbayev.kz).
- **[KazQAD](https://github.com/IS2AI/KazQAD)** — источник обучающих пассажей (казахская
  Википедия, CC BY-SA 4.0).

## Пайплайн

| шаг | скрипт |
|---|---|
| выбор базы (zero-shot) | `scripts/zeroshot_107m.py` |
| KazQAD → пары + KazQAD-негативы (rel=0) | `scripts/prepare_data.py` |
| антилик против бенчмарка | `scripts/check_overlap.py` |
| синтетика (57K пар) | `scripts/generate_synthetic.py` |
| hard-negatives BM25 + казахский стеммер | `scripts/mine_hard_negatives.py` |
| сборка train-файла (id → тексты) | `scripts/build_train_v2.py` |
| обучение (T4, CachedMNRL, seq 512) | `scripts/train.py` |
| оценка + гибрид с BM25(стеммер) | `scripts/eval.py` |
| оценка (OOD: речи) | `scripts/eval_ood.py` |

Готовые Colab/Kaggle-ноутбуки прогонов — в `notebooks/`. Казахский стеммер (для
hard-negatives и BM25-канала гибрида, ключ `KAZAKH_STEMMER_KEY`) — доступ на
[qaz-api.vercel.app](https://qaz-api.vercel.app/).

**v2 обучена на:** 57 369 синтетических пар (казвики) + по 1 hard-negative на пару
(намайнен BM25 с **казахским стеммером**) + 3 733 KazQAD gold (с его rel=0 негативами)
= 61 102 обучающих примера. 278m, CachedMNRL, 2 эпохи, lr 1e-5, **max_seq_len 512**,
Kaggle T4. Отличие v2 от v1 — именно стеммер-негативы (усиление морфологии) и обучение
на 512.

## Данные и лицензия

`data/synthetic_pairs.jsonl` — 57 369 пар (после антилика), синтетика на пассажах
**KazQAD** (казахская Википедия, **CC BY-SA 4.0**, атрибуция KazQAD). Компактные
артефакты негативов — `data/synthetic_pairs.hn.ids.jsonl`, KazQAD gold —
`data/kazqad_pairs.dedup.jsonl`. Полные результаты — в `results/`.

Модель: [`Tim2190/granite-278m-kk`](https://huggingface.co/Tim2190/granite-278m-kk)
(fp16, ~556 МБ; v2 в `main`, v1 под тегом `v1`).
