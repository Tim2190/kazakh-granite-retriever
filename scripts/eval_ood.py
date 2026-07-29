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
    ap.add_argument("--bm25-stemmer", choices=["identity", "kazakh", "kazakh-prod"],
                    default="identity",
                    help="kazakh — demo-сервис; kazakh-prod — production (KAZAKH_STEMMER_KEY).")
    ap.add_argument("--rrf-k", type=int, default=60, help="Константа RRF для гибрида.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.benchmark_root).resolve()))
    from src.retrieval.dense import DenseIndex
    from src.retrieval.bm25 import BM25Index, default_analyzer
    from src.preprocess.stemmer import get_stemmer
    from src.eval import metrics

    data = Path(args.ood_root) / "data"
    corpus = load_passages(data / "passages.jsonl")
    qmap, cats = load_queries(data / "queries.jsonl")
    qrels = load_qrels(data / "qrels.jsonl")
    print(f"OOD: пассажей {len(corpus)}, запросов {len(qmap)}, с qrels {len(qrels)}")
    print(f"тиры: {dict((t, sum(1 for c in cats.values() if c == t)) for t in sorted(set(cats.values())))}")

    DEPTH = 100  # глубина выдачи для RRF-гибрида (корпус мал, дёшево)

    def run_dense(model_key: str) -> Dict[str, List[str]]:
        print(f"\n>>> Dense: {model_key}")
        idx = DenseIndex(model_name=model_key, query_prefix="", doc_prefix="",
                         batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        idx.index(corpus)
        return idx.run(qmap, top_k=DEPTH)

    if args.bm25_stemmer == "kazakh-prod":
        from mine_hard_negatives import KazakhStemmerProd
        from src.preprocess.tokenize import tokenize
        st = KazakhStemmerProd(cache_path="results/stem_cache.json")
        uniq = set()
        for _, t in corpus:
            uniq.update(tokenize(t))
        for t in qmap.values():
            uniq.update(tokenize(t))
        print(f"\n>>> Прогрев казахского стеммера (prod): {len(uniq):,} уникальных слов…")
        st.warm(uniq)
        analyzer = default_analyzer(st)
    else:
        analyzer = default_analyzer(get_stemmer(args.bm25_stemmer))
    print(f">>> BM25 (стеммер={args.bm25_stemmer})")
    bm = BM25Index(analyzer=analyzer).index(corpus)
    run_bm25 = bm.run(qmap, top_k=DEPTH)
    run_base = run_dense(args.base_model)
    run_ft = run_dense(args.finetuned)

    def rrf(runs: List[Dict[str, List[str]]], k: int, top_k: int = 10) -> Dict[str, List[str]]:
        fused: Dict[str, List[str]] = {}
        qids = set().union(*[set(r) for r in runs])
        for q in qids:
            score: Dict[str, float] = {}
            for r in runs:
                for rank, doc in enumerate(r.get(q, []), start=1):
                    score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
            fused[q] = [d for d, _ in sorted(score.items(), key=lambda x: -x[1])[:top_k]]
        return fused

    run_hyb = rrf([run_ft, run_bm25], k=args.rrf_k, top_k=10)

    systems = {"BM25": run_bm25, "Granite-zs": run_base,
               "Granite-ft": run_ft, "FT⊕BM25": run_hyb}
    tiers = sorted(set(cats.values()))
    rows = []

    def ndcg(run, sub):
        return metrics.evaluate_run(run, sub, metrics=("ndcg",), ks=(10,))["ndcg@10"]

    def block(name: str, sub: Dict[str, Set[str]]):
        row = {"scope": name}
        for sysname, run in systems.items():
            row[sysname] = round(ndcg(run, sub), 4)
        _, p, _ = metrics.paired_bootstrap(run_base, run_ft, sub, metric="ndcg",
                                           k=10, n_resamples=args.n_resamples)
        row["p_ft_vs_zs"] = round(p, 4)
        rows.append(row)

    block("ALL", qrels)
    for t in tiers:
        qids = {q for q, c in cats.items() if c == t}
        block(t, {q: rel for q, rel in qrels.items() if q in qids})

    print("\n=== OOD (речи): nDCG@10 по системам ===")
    hdr = f"{'scope':<14}" + "".join(f"{s:>12}" for s in systems) + f"{'p(ft>zs)':>10}"
    print(hdr)
    for r in rows:
        line = f"{r['scope']:<14}" + "".join(f"{r[s]:>12.3f}" for s in systems)
        sig = "*" if r["p_ft_vs_zs"] < 0.05 else " "
        print(line + f"{r['p_ft_vs_zs']:>9.3f}{sig}")
    print("  * = прирост ft над zero-shot значим (p<0.05). BM25-стеммер:", args.bm25_stemmer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "RAG-Two-Pass-Retrieval-QAZ (OOD, речи)",
               "base_model": args.base_model, "finetuned": args.finetuned,
               "bm25_stemmer": args.bm25_stemmer,
               "n_passages": len(corpus), "n_queries": len(qmap), "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")


if __name__ == "__main__":
    main()
