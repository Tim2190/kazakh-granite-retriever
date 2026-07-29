#!/usr/bin/env python3
"""
Шаг е-2 — независимая OOD-проверка на RAG-Two-Pass-Retrieval-QAZ.

Второй, независимый бенчмарк (github.com/Tim2190/RAG-Two-Pass-Retrieval-QAZ):
471 пассаж из официальных речей (akorda.kz / nazarbayev.kz) — **другой домен**,
не Википедия. Наша модель на нём НИКОГДА не обучалась → честная проверка на
обобщение, а не переобучение под основной бенчмарк.

Данные (data/ в клоне того репо):
    passages.jsonl : {id, text, title, source, ...}
    queries.jsonl  : {query_id, query, type∈{factoid,paraphrase,low_overlap}, ...}
    qrels.jsonl    : {query_id, passage_id, relevance}

Метрики и энкодинг переиспользуем из основного бенчмарка (src.retrieval.dense +
src.eval.metrics), чтобы цифры были сопоставимы с ID-замером. Granite → без
префиксов (как везде). Считаем ALL + разбивку по тирам + paired bootstrap.

Запуск:
    python scripts/eval_ood.py \\
        --benchmark-root /content/bench \\        # Kaz-RAG (нужен ради DenseIndex/metrics)
        --ood-root /content/RAG-Two-Pass-Retrieval-QAZ \\
        --base-model ibm-granite/granite-embedding-278m-multilingual \\
        --finetuned /path/or/hf/granite-278m-40k \\
        --max-seq-len 256 \\
        --out results/eval_ood_278m_40k.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_passages(path: Path) -> List[Tuple[str, str]]:
    out = []
    for o in _jsonl(path):
        did = o.get("id") or o.get("passage_id") or o.get("_id")
        text = o.get("text") or o.get("passage") or ""
        if did and text:
            out.append((str(did), text))
    return out


def load_queries(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    qmap, cats = {}, {}
    for o in _jsonl(path):
        qid = o.get("query_id") or o.get("id")
        qtext = o.get("query") or o.get("text")
        if qid and qtext:
            qmap[str(qid)] = qtext
            cats[str(qid)] = o.get("type", "?")
    return qmap, cats


def load_qrels(path: Path) -> Dict[str, Set[str]]:
    d: Dict[str, Set[str]] = {}
    for o in _jsonl(path):
        try:
            rel = float(o.get("relevance", 0))
        except (TypeError, ValueError):
            rel = 0
        if rel > 0:
            d.setdefault(str(o["query_id"]), set()).add(str(o["passage_id"]))
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD-замер на RAG-Two-Pass-Retrieval-QAZ")
    ap.add_argument("--benchmark-root", required=True,
                    help="Клон Kaz-RAG-search-benchmark (ради src.retrieval.dense + src.eval.metrics).")
    ap.add_argument("--ood-root", required=True,
                    help="Клон RAG-Two-Pass-Retrieval-QAZ (там data/).")
    ap.add_argument("--base-model", default="ibm-granite/granite-embedding-278m-multilingual")
    ap.add_argument("--finetuned", required=True, help="Путь или HF id дообученной модели.")
    ap.add_argument("--out", default="results/eval_ood.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--n-resamples", type=int, default=10000)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.benchmark_root).resolve()))
    from src.retrieval.dense import DenseIndex
    from src.eval import metrics

    data = Path(args.ood_root) / "data"
    corpus = load_passages(data / "passages.jsonl")
    qmap, cats = load_queries(data / "queries.jsonl")
    qrels = load_qrels(data / "qrels.jsonl")
    print(f"OOD: пассажей {len(corpus)}, запросов {len(qmap)}, с qrels {len(qrels)}")
    print(f"тиры: {dict((t, sum(1 for c in cats.values() if c == t)) for t in sorted(set(cats.values())))}")

    def run_model(model_key: str) -> Dict[str, List[str]]:
        print(f"\n>>> Прогон: {model_key}")
        idx = DenseIndex(model_name=model_key, query_prefix="", doc_prefix="",
                         batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        idx.index(corpus)
        return idx.run(qmap, top_k=max(args.top_k, 10))

    run_base = run_model(args.base_model)
    run_ft = run_model(args.finetuned)

    tiers = sorted(set(cats.values()))
    rows = []

    def block(name: str, sub: Dict[str, Set[str]]):
        base = metrics.evaluate_run(run_base, sub, metrics=("ndcg", "mrr", "recall"), ks=(10,))
        ft = metrics.evaluate_run(run_ft, sub, metrics=("ndcg", "mrr", "recall"), ks=(10,))
        delta, p, _ = metrics.paired_bootstrap(run_base, run_ft, sub, metric="ndcg",
                                               k=10, n_resamples=args.n_resamples)
        rows.append({"scope": name,
                     "ndcg@10_zeroshot": round(base["ndcg@10"], 4),
                     "ndcg@10_finetuned": round(ft["ndcg@10"], 4),
                     "delta": round(delta, 4), "p_value": round(p, 4),
                     "recall@10_zeroshot": round(base["recall@10"], 4),
                     "recall@10_finetuned": round(ft["recall@10"], 4)})

    block("ALL", qrels)
    for t in tiers:
        qids = {q for q, c in cats.items() if c == t}
        block(t, {q: rel for q, rel in qrels.items() if q in qids})

    print("\n=== OOD (речи): Zero-shot vs Fine-tuned (nDCG@10) ===")
    print(f"{'scope':<16}{'zero-shot':>10}{'fine-tuned':>12}{'Δ':>9}{'p':>8}")
    for r in rows:
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"{r['scope']:<16}{r['ndcg@10_zeroshot']:>10.3f}{r['ndcg@10_finetuned']:>12.3f}"
              f"{r['delta']:>+9.3f}{r['p_value']:>8.3f}{sig}")
    print("  * = значимо (p<0.05, paired bootstrap). Δ = fine-tuned − zero-shot.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "RAG-Two-Pass-Retrieval-QAZ (OOD, речи)",
               "base_model": args.base_model, "finetuned": args.finetuned,
               "n_passages": len(corpus), "n_queries": len(qmap), "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")


if __name__ == "__main__":
    main()
