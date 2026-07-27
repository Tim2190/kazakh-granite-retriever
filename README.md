# kazakh-granite-retriever

Дообучение IBM Granite-embedding под казахский retrieval.

## Тезис

Официальный список поддерживаемых языков ≠ реальная применимость. Granite-embedding
**R1** (107m / 278m multilingual) казахский официально **не** заявляет, но эмпирически
на [Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
держит казахский не хуже официально поддержанной **R2** (у которой токенизатор дробит
слова в ~2.3× сильнее). Цель проекта — показать, что таргетированный файнтюн на
казахском retrieval закрывает то, чего не было в исходном обучении.

Базовая модель под файнтюн: **`ibm-granite/granite-embedding-107m-multilingual` (R1)**
— эффективный вариант (меньше, быстрее, дешевле в деплое).

## Что уже известно из бенчмарка (зеро-шот, nDCG@10)

| Модель | inflected | natural | **vocab-gap** | ALL |
|---|---|---|---|---|
| Granite-278m (R1) | 0.791 | 0.923 | **0.303** | 0.672 |
| Granite-311m (R2) | 0.791 | 0.924 | **0.263** | 0.659 |
| **Granite-107m (R1)** ← база под файнтюн | 0.732 | 0.876 | **0.242** | **0.617** |
| Granite-97m (R2) | 0.711 | 0.880 | **0.175** | 0.589 |
| multilingual-e5-base | 0.845 | 0.947 | 0.562 | 0.785 |
| kazakh-e5 (shyngys879) | 0.836 | 0.909 | 0.497 | 0.747 |
| BM25 + stemmer | 0.727 | 0.772 | 0.764 | 0.754 |
| kazakh-e5 ⊕ BM25+st | **0.862** | 0.869 | 0.694 | **0.808** |

Ключевой вывод: слабое место Granite — **не морфология, а семантика** (vocab-gap
проваливается до 0.17–0.30). Это и есть главная мишень файнтюна. Планка «победы» —
обойти пуре-денс kazakh-e5 (0.747) и приблизиться к гибриду (0.808), поэтому
гибрид `Granite-ft ⊕ BM25` заложен в цель, а не как бонус.

**Итог Шага 0 (зеро-шот 107m R1, `scripts/zeroshot_107m.py` через Colab):**
107m R1 даёт **ALL nDCG@10 = 0.617** и обходит младшую официально-казахскую
97m R2 (0.589) — то есть даже на маленьком tier R1-без-казахского ≥ R2-с-казахским.
Слабость подтвердилась ровно на vocab-gap (0.242). Стартовая точка файнтюна: **0.617**,
цель — тянуть vocab-gap вверх и в сумме превзойти kazakh-e5 (0.747).

## Пайплайн

| Шаг | Скрипт | Статус |
|---|---|---|
| **0.** Зеро-шот базы по фактам | `scripts/zeroshot_107m.py` | ✅ готов |
| **в.** KazQAD → query/positive + hard-neg (rel=0) | `scripts/prepare_data.py` | ✅ готов |
| **г.** Дедуп до обучения (title + near-dup + article_id) | `scripts/check_overlap.py` | ✅ готов |
| **д.** Обучение (Kaggle T4, CachedMNRL) | `scripts/train.py` | ✅ готов |
| **е.** Оценка zero-shot vs fine-tuned + significance | `scripts/eval.py` | ✅ готов |

## Шаг 0 — запуск

Клонируй бенчмарк рядом и поставь зависимости:

```bash
git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark ../Kaz-RAG-search-benchmark
pip install -r requirements.txt
```

Прогон зеро-шота 107m R1 (тем же харнессом `src.eval.run_dense`, что и 278m):

```bash
python scripts/zeroshot_107m.py --benchmark-root ../Kaz-RAG-search-benchmark \
                                --out results/zeroshot_107m.json
```

Скрипт вызывает штатный `run_dense.run(...)` бенчмарка с HF-id 107m. За счёт фолбэка
`MODELS.get(model_key, (model_key, "", ""))` модель идёт с **пустыми префиксами** —
идентично пути granite-278m R1, так что цифры сравнимы по построению. На выходе —
таблица метрик, JSON в `results/` и строка 107m рядом с референсами линейки.

## Данные

- **KazQAD** ([IS2AI](https://github.com/IS2AI/KazQAD), CC BY-SA) — 800K+ пассажей
  казахской Википедии + ~61K троек question–passage–answer.
- **Дедупликация обязательна.** Оба корпуса восходят к казахской Википедии, поэтому
  пересечение почти гарантировано. `check_overlap.py` чистит в два слоя: title/ID-матч
  + near-dup на уровне пассажей — иначе train/test leakage обнулит результат.
