# data/

Тренировочные данные для дообучения (в git трекается только синтетика — крупные
исходники KazQAD/бенчмарка не коммитятся, они регенерируются скриптами).

## synthetic_pairs.jsonl

Синтетические пары (query → passage) для казахского retrieval, сгенерированы
`scripts/generate_synthetic.py` (Gemini) на пассажах корпуса **KazQAD**.
Три категории: `natural`, `vocab_gap`, `inflected`. Статьи оценочного бенчмарка
исключены по article_id (антилик).

**Источник пассажей:** KazQAD (github.com/IS2AI/KazQAD), казахская Википедия,
лицензия **CC BY-SA 4.0**. Производные пары наследуют CC BY-SA 4.0 с атрибуцией
KazQAD / Kazakh Wikipedia. Вопросы сгенерированы автоматически (Gemini).
