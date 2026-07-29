#!/usr/bin/env python3
"""
Шаг д — дообучение Granite-107m R1 на очищенном KazQAD (Kaggle T4).

База: ibm-granite/granite-embedding-107m-multilingual (R1). Зеро-шот на бенчмарке
= 0.617 nDCG@10, слабое место — vocab-gap (0.242). Дообучаем retrieval так, чтобы
поднять именно семантику.

Данные: выход check_overlap.py (уже без пересечений с бенчмарком!). Формат строки:
    {"query", "positive", "negatives":[...], ...}
Из него строим:
  • триплеты (anchor, positive, negative) — по одному на каждый hard-negative (rel=0
    из KazQAD): жёсткие негативы бьют прямо в vocab-gap;
  • пары (anchor, positive) — для троек без hard-negatives (работают на in-batch
    негативах).
Обе части учим одним лоссом CachedMultipleNegativesRankingLoss (GradCache): он даёт
большой эффективный батч на 16 ГБ T4 — а качество MNRL прямо растёт с числом
негативов. Если версия sentence-transformers старая — фолбэк на обычный MNRL.

ВАЖНО (честность к eval): Granite R1 работает БЕЗ query/passage-префиксов — ровно
как в eval-харнессе бенчмарка. Никаких "query:"/"passage:" здесь не добавляем.

Запуск (Kaggle, GPU on):
    python scripts/train.py \\
        --data data/kazqad_pairs.dedup.jsonl \\
        --output-dir models/granite-107m-kk \\
        --epochs 2 --batch-size 128 --mini-batch 32 --lr 2e-5

Дефолты подобраны под T4; финально подкрутим после того, как узнаем размер
очищенного датасета (report из check_overlap).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_examples(paths, max_neg_per_pair: int
                  ) -> Tuple[List[dict], List[dict]]:
    """Один или несколько JSONL → (триплеты[a,p,n], пары[a,p]).
    Триплеты разворачиваются по негативам; строки без негативов идут в пары."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    triplets: List[dict] = []
    pairs: List[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                q, pos = r.get("query"), r.get("positive")
                if not q or not pos:
                    continue
                negs = [n for n in (r.get("negatives") or []) if n][:max_neg_per_pair]
                if negs:
                    for n in negs:
                        triplets.append({"anchor": q, "positive": pos, "negative": n})
                else:
                    pairs.append({"anchor": q, "positive": pos})
    return triplets, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Granite-107m R1 на KazQAD (dedup)")
    ap.add_argument("--data", required=True, nargs="+",
                    help="Один или несколько JSONL (напр. синтетика + KazQAD dedup).")
    ap.add_argument("--base-model", default="ibm-granite/granite-embedding-107m-multilingual")
    ap.add_argument("--output-dir", default="models/granite-107m-kk")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Эффективный батч (число негативов в MNRL). GradCache держит память.")
    ap.add_argument("--mini-batch", type=int, default=32,
                    help="Мини-батч GradCache (реальный forward). Уменьшай при OOM.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-neg-per-pair", type=int, default=4,
                    help="Сколько hard-negatives на тройку разворачивать.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    # По умолчанию обучаем в fp32: надёжно (Granite грузится в bf16, а fp16-AMP
    # GradScaler не умеет разгребать bf16-градиенты → краш). На 107m/T4 fp32 быстр.
    ap.add_argument("--fp16", action="store_true",
                    help="fp16 AMP (быстрее на T4). Модель кастуется в fp32, так что краша нет.")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 (Ampere+; на T4 не рекомендуется).")
    ap.add_argument("--save-fp32", action="store_true",
                    help="Сохранить модель в fp32 (~1.1 ГБ). По умолчанию fp16 (~556 МБ).")
    ap.add_argument("--push-to-hub", default=None,
                    help="Repo id на HF Hub для пуша (опционально, нужен HF_TOKEN).")
    args = ap.parse_args()

    random.seed(args.seed)
    from datasets import Dataset
    import torch
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments, losses,
    )
    torch.manual_seed(args.seed)

    triplets, pairs = load_examples([Path(p) for p in args.data], args.max_neg_per_pair)
    print(f"Данные: {', '.join(args.data)}")
    print(f"Триплетов (a,p,hard-neg): {len(triplets):,} | пар (a,p): {len(pairs):,}")
    if not triplets and not pairs:
        raise SystemExit("Пустой датасет — проверь --data.")

    train_dataset: Dict[str, Dataset] = {}
    if triplets:
        train_dataset["triplet"] = Dataset.from_list(triplets)
    if pairs:
        train_dataset["pair"] = Dataset.from_list(pairs)

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_len
    # Granite отдаёт веса в bf16 — приводим к fp32, иначе fp16-AMP падает на unscale.
    model = model.to(torch.float32)

    # CachedMNRL (GradCache) — большой эффективный батч на T4; фолбэк для старых ST.
    try:
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
        print(f"Лосс: CachedMultipleNegativesRankingLoss (mini_batch={args.mini_batch})")
    except AttributeError:
        loss = losses.MultipleNegativesRankingLoss(model)
        print("Лосс: MultipleNegativesRankingLoss (CachedMNRL недоступен — обнови sentence-transformers)")

    n_examples = len(triplets) + len(pairs)
    steps_per_epoch = max(1, math.ceil(n_examples / args.batch_size))
    print(f"~{steps_per_epoch} шагов/эпоху, эпох={args.epochs}")

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="no",   # без промежуточных чекпойнтов с optimizer state
        seed=args.seed,       # (иначе папка модели раздувается на ~3 ГБ Adam-состояния)
        dataloader_drop_last=True,     # ровные батчи → корректные in-batch негативы
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset if len(train_dataset) > 1 else next(iter(train_dataset.values())),
        loss=loss,
    )
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.save_fp32:
        model = model.half()          # fp16 для инференса → ~556 МБ вместо ~1.1 ГБ (fp32)
    model.save(str(out))
    dtype = "fp32" if args.save_fp32 else "fp16"
    print(f"\nМодель сохранена ({dtype}) → {out}")
    print("Дальше: eval.py — прогон на бенчмарке + paired bootstrap vs зеро-шот.")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        print(f"Запушено на HF Hub → {args.push_to_hub}")


if __name__ == "__main__":
    main()
