# data/

Тренировочные данные для дообучения (в git трекается только синтетика — крупные
исходники KazQAD/бенчмарка не коммитятся, они регенерируются скриптами).

## synthetic_pairs.jsonl

Синтетические пары (query → passage) для казахского retrieval, сгенерированы
`scripts/generate_synthetic.py` (Gemini) на пассажах корпуса **KazQAD**.
Три категории: `natural`, `vocab_gap`, `inflected` (по одному вопросу каждого типа
на пассаж). Статьи оценочного бенчмарка исключены по article_id (антилик).

**Готово: 57 369 пар** (после антилика), баланс категорий ~1:1:1.

## synthetic_pairs.hn.ids.jsonl

Компактный артефакт hard-negatives: `query` + `positive_id` + `negative_ids` (без
полных текстов — они разворачиваются из корпуса `scripts/build_train_v2.py`).
Негативы намайнены `scripts/mine_hard_negatives.py` через BM25 с казахским стеммером.

## kazqad_pairs.dedup.jsonl

Размеченные KazQAD gold-пары (3 733 после дедупа против бенчмарка), с rel=0 негативами.

## exclude_article_ids.txt

article_id статей оценочного бенчмарка — исключаются при генерации/майнинге (антилик).

**Источник пассажей:** KazQAD (github.com/IS2AI/KazQAD), казахская Википедия,
лицензия **CC BY-SA 4.0**. Производные пары наследуют CC BY-SA 4.0 с атрибуцией
KazQAD / Kazakh Wikipedia. Вопросы сгенерированы автоматически (Gemini).
