# kazakh-granite-retriever

*English — [README_EN.md](README_EN.md).*

**Granite-278m-kk** — компактная (278M) embedding-модель для казахского поиска и RAG,
дообученная из `ibm-granite/granite-embedding-278m-multilingual` (R1). Казахского в
официальном списке Granite нет — модель показывает, что таргетированный файнтюн
закрывает это на практике.

> **📄 Полный отчёт — [REPORT.md](REPORT.md)** (метод, все таблицы, выводы).

## Итог

- Обходит специализированные казахские файнтюны (kazakh-e5, kazembed-v5) на двух
  независимых доменах.
- Гибрид с BM25 (**0.814** nDCG@10) обходит референс kazakh-e5 ⊕ BM25 (0.808) и
  базовый e5-base (0.785) на основном бенчмарке.
- Улучшает базовый Granite-R1 значимо (0.672 → 0.752 dense / 0.814 hybrid).
- Прирост подтверждён на независимом OOD-домене (официальные речи) — значимо по всем срезам.

**Основной бенчмарк (Википедия), nDCG@10 (ALL):**

| # | система | ALL |
|---|---|---|
| 1 | bge-m3 | 0.866 |
| 2 | jina-v3 | 0.821 |
| **3** | **Granite-278m-kk ⊕ BM25 (наш)** | **0.814** |
| 4 | kazakh-e5 ⊕ BM25 | 0.808 |
| 6 | multilingual-e5-base | 0.785 |
| **8** | **Granite-278m-kk (наш, dense)** | **0.752** |
| 9 | kazakh-e5 | 0.747 |

Верхний сегмент: dense обходит специализированную kazakh-e5, гибрид — её связку с
BM25 и e5-base. Сильные общие модели (bge-m3, jina-v3) впереди — ориентир для развития.

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
| синтетика (40K пар) | `scripts/generate_synthetic.py` |
| обучение (T4, CachedMNRL) — синтетика + KazQAD gold | `scripts/train.py` |
| оценка + гибрид с BM25(казахский стеммер) | `scripts/eval.py` |
| оценка (OOD: речи) | `scripts/eval_ood.py` |
| _(абляция, не в финале)_ BM25-майнинг hard-neg | `scripts/mine_hard_negatives.py` |

Финальная модель обучена на **синтетике (40K) + KazQAD gold** (с его негативами `rel=0`).
Казахский стеммер задействован **только в BM25-канале гибрида** на этапе оценки.
BM25-майнинг hard-negatives (`mine_hard_negatives.py`) исследован как абляция — прироста
не дал, в финальную модель не вошёл.

## Данные и лицензия

`data/synthetic_pairs.jsonl` — 40 084 пары, синтетика на пассажах **KazQAD** (казахская
Википедия, **CC BY-SA 4.0**, атрибуция KazQAD). Полные результаты — в `results/`.

Модель будет опубликована на HuggingFace (ссылка добавится).
