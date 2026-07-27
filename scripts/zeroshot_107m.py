#!/usr/bin/env python3
"""
Шаг 0 — зеро-шот Granite-embedding-107m-multilingual (R1) на Kaz-RAG-search-benchmark.

107m R1 — единственная непротестированная модель линейки Granite в бенчмарке
(278m R1 и 97m/311m R2 уже есть в results/SPRINT2_NEW_MODELS.md). Этот скрипт
прогоняет её зеро-шот и печатает строку для сравнительной таблицы, чтобы финальную
базу под файнтюн выбрать по цифрам, а не по документации.

ВАЖНО про честность сравнения:
    Мы НЕ переписываем метрики/энкодинг. Скрипт импортирует и вызывает штатный
    харнесс бенчмарка `src.eval.run_dense.run(...)`. В нём есть фолбэк
    `MODELS.get(model_key, (model_key, "", ""))`, поэтому переданный HF-id идёт
    с ПУСТЫМИ префиксами (как и granite-278m R1) через тот же `DenseIndex` и те же
    функции из `src.eval.metrics`. То есть 107m проходит буквально тем же путём,
    что 278m — сравнение корректно по построению.

Запуск (репозитории лежат рядом):
    kazakh-granite-retriever/
    Kaz-RAG-search-benchmark/

    python scripts/zeroshot_107m.py \
        --benchmark-root ../Kaz-RAG-search-benchmark \
        --out results/zeroshot_107m.json

Или через переменную окружения:
    export KAZ_RAG_BENCHMARK=../Kaz-RAG-search-benchmark
    python scripts/zeroshot_107m.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

MODEL_ID = "ibm-granite/granite-embedding-107m-multilingual"  # R1, официально без казахского

# Референсные nDCG@10 (ALL) из results/SPRINT2_NEW_MODELS.md — для быстрого контекста.
# Печатаются рядом с полученной цифрой 107m; это НЕ пересчёт, а справочные значения.
REFERENCE_NDCG10_ALL = {
    "Granite-278m (R1)": 0.672,
    "Granite-311m (R2)": 0.659,
    "Granite-97m  (R2)": 0.589,
    "multilingual-e5-base": 0.785,
    "kazakh-e5 (shyngys879)": 0.747,
    "BM25 + stemmer": 0.754,
}


def _locate_benchmark(cli_value: str | None) -> Path:
    """Найти корень Kaz-RAG-search-benchmark (где лежит пакет `src`)."""
    candidates = [
        cli_value,
        os.environ.get("KAZ_RAG_BENCHMARK"),
        "../Kaz-RAG-search-benchmark",
        "./Kaz-RAG-search-benchmark",
    ]
    for c in candidates:
        if not c:
            continue
        root = Path(c).expanduser().resolve()
        if (root / "src" / "eval" / "run_dense.py").exists():
            return root
    raise SystemExit(
        "Не найден корень бенчмарка (папка с src/eval/run_dense.py).\n"
        "Укажи --benchmark-root <путь> или переменную окружения KAZ_RAG_BENCHMARK.\n"
        "Клонируй: git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Зеро-шот Granite-107m R1 на Kaz-RAG-search-benchmark (тот же харнесс)."
    )
    ap.add_argument("--benchmark-root", default=None,
                    help="Путь к клонированному Kaz-RAG-search-benchmark "
                         "(иначе KAZ_RAG_BENCHMARK или ../Kaz-RAG-search-benchmark).")
    ap.add_argument("--corpus", default=None,
                    help="corpus.jsonl (по умолчанию <benchmark>/data/corpus/corpus.jsonl).")
    ap.add_argument("--queries", default=None,
                    help="queries.jsonl (по умолчанию <benchmark>/data/queries/queries.jsonl).")
    ap.add_argument("--model-id", default=MODEL_ID,
                    help=f"HF id базовой модели (по умолчанию {MODEL_ID}).")
    ap.add_argument("--out", default="results/zeroshot_107m.json",
                    help="Куда сохранить метрики (JSON).")
    ap.add_argument("--emb-cache", default="results/emb_granite_107m",
                    help="Префикс кэша эмбеддингов корпуса.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    bench_root = _locate_benchmark(args.benchmark_root)
    # Кэш эмбеддингов бенчмарк пишет относительно cwd — считаем от корня бенчмарка,
    # чтобы форматы путей/данных совпадали с эталонными прогонами.
    sys.path.insert(0, str(bench_root))
    corpus = args.corpus or str(bench_root / "data" / "corpus" / "corpus.jsonl")
    queries = args.queries or str(bench_root / "data" / "queries" / "queries.jsonl")

    from src.eval import run_dense          # noqa: E402  (после правки sys.path)
    from src.eval.run_benchmark import _fmt_table  # noqa: E402

    print(f"Бенчмарк:  {bench_root}")
    print(f"Модель:    {args.model_id}  (R1, зеро-шот, пустые префиксы — как granite-278m)")
    print(f"Корпус:    {corpus}")
    print(f"Запросы:   {queries}\n")

    result = run_dense.run(
        corpus_path=corpus,
        queries_path=queries,
        model_key=args.model_id,       # HF id → фолбэк даёт пустые префиксы
        emb_cache=args.emb_cache,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    print(_fmt_table(result))

    ndcg10_all = result["overall"].get("ndcg@10")
    print("\n=== nDCG@10 (ALL): 107m R1 в контексте линейки (референс — SPRINT2) ===")
    rows = dict(REFERENCE_NDCG10_ALL)
    rows[">>> Granite-107m (R1) [этот прогон]"] = round(ndcg10_all, 3) if ndcg10_all else None
    for name, val in sorted(rows.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        mark = "  <-- 107m" if name.startswith(">>>") else ""
        print(f"  {name:<34} {val:.3f}{mark}" if val is not None else f"  {name:<34}   n/a")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact = {k: v for k, v in result.items() if k != "run"}
    compact["_meta"] = {
        "step": "0-zeroshot",
        "model_id": args.model_id,
        "benchmark_root": str(bench_root),
        "seed": args.seed,
        "reference_ndcg10_all": REFERENCE_NDCG10_ALL,
    }
    json.dump(compact, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены → {out_path}")


if __name__ == "__main__":
    main()
