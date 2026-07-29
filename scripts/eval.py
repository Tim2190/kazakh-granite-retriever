#!/usr/bin/env python3
"""
Шаг е — оценка: зеро-шот Granite-107m R1 vs дообученная модель на
Kaz-RAG-search-benchmark, теми же метриками и харнессом, + статзначимость.

Как и на Шаге 0, НЕ переписываем метрики: гоняем обе модели через штатный
`src.eval.run_dense.run` бенчмарка (пустые префиксы — путь Granite R1), берём
per-query ранжировки и считаем `src.eval.metrics.paired_bootstrap` (тот самый
10k-resamples тест, что уже в бенчмарке). Разбивка по категориям показывает,
где именно файнтюн сработал — ожидаем главный прирост на vocabulary-gap
(зеро-шот 107m там всего 0.242).

Запуск (Colab, GPU on):
    python scripts/eval.py \\
        --benchmark-root /content/Kaz-RAG-search-benchmark \\
        --finetuned models/granite-107m-kk \\
        --out results/eval_107m_ft_vs_zeroshot.json

Гибрид ⊕BM25 (на этом бенчмарке гибрид — сильнейший) считается штатным
`python -m src.eval.run_hybrid` бенчмарка на ранжировках дообученной модели —
см. подсказку в конце вывода.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _locate_benchmark(cli_value):
    import os
    for c in (cli_value, os.environ.get("KAZ_RAG_BENCHMARK"),
              "../Kaz-RAG-search-benchmark", "./Kaz-RAG-search-benchmark"):
        if not c:
            continue
        root = Path(c).expanduser().resolve()
        if (root / "src" / "eval" / "run_dense.py").exists():
            return root
    raise SystemExit("Не найден бенчмарк — укажи --benchmark-root или KAZ_RAG_BENCHMARK.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot vs fine-tuned на Kaz-RAG-search-benchmark")
    ap.add_argument("--benchmark-root", default=None)
    ap.add_argument("--finetuned", required=True,
                    help="Путь (или HF id) дообученной модели.")
    ap.add_argument("--base-model", default="ibm-granite/granite-embedding-107m-multilingual",
                    help="Базовая модель для зеро-шот сравнения.")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--out", default="results/eval_107m_ft_vs_zeroshot.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--hybrid", action="store_true",
                    help="Также посчитать BM25 и гибрид FT⊕BM25 (RRF).")
    ap.add_argument("--bm25-stemmer", choices=["identity", "kazakh", "kazakh-prod"],
                    default="identity",
                    help="kazakh — demo-сервис (медленно); kazakh-prod — production "
                         "(X-API-Key из KAZAKH_STEMMER_KEY, быстрый прогрев).")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--vs-model", default=None,
                    help="Доп. модель для paired-сравнения с дообученной (напр. shyngys-e5 "
                         "— короткое имя из бенчмарка, чтобы применились e5-префиксы).")
    args = ap.parse_args()

    bench_root = _locate_benchmark(args.benchmark_root)
    sys.path.insert(0, str(bench_root))
    corpus = args.corpus or str(bench_root / "data" / "corpus" / "corpus.jsonl")
    queries = args.queries or str(bench_root / "data" / "queries" / "queries.jsonl")

    from src.eval import run_dense, metrics
    from src.queries import dataset

    q = dataset.load_queries(queries)
    qrels = dataset.qrels_from_queries(q)
    cats = dataset.categories_from_queries(q)

    DEPTH = 100 if args.hybrid else max(args.top_k, 10)

    def run_model(model_key, cache):
        print(f"\n>>> Прогон: {model_key}")
        return run_dense.run(corpus, queries, model_key, emb_cache=cache,
                             top_k=DEPTH, batch_size=args.batch_size,
                             max_seq_len=args.max_seq_len)

    def cache_key(m: str) -> str:
        # уникальный кэш эмбеддингов на КАЖДУЮ модель (иначе 278m подтянет 107m-кэш)
        return "results/emb_" + re.sub(r"[^0-9A-Za-z]+", "_", m).strip("_")[-60:]

    res_base = run_model(args.base_model, cache_key(args.base_model))
    res_ft = run_model(args.finetuned, cache_key(args.finetuned) + "_ft")
    run_base, run_ft = res_base["run"], res_ft["run"]
    run_vs = run_model(args.vs_model, cache_key(args.vs_model))["run"] if args.vs_model else None

    run_bm25 = run_hyb = None
    if args.hybrid:
        from src.retrieval.bm25 import BM25Index, default_analyzer
        corpus_pairs = dataset.load_corpus(corpus)
        qmap_bm = dataset.queries_as_map(q)
        if args.bm25_stemmer == "kazakh-prod":
            from mine_hard_negatives import KazakhStemmerProd
            from src.preprocess.tokenize import tokenize
            stemmer = KazakhStemmerProd(cache_path="results/stem_cache.json")
            uniq = set()
            for _, t in corpus_pairs:
                uniq.update(tokenize(t))
            for t in qmap_bm.values():
                uniq.update(tokenize(t))
            print(f"\n>>> Прогрев казахского стеммера (prod): {len(uniq):,} уникальных слов…")
            stemmer.warm(uniq)
            analyzer = default_analyzer(stemmer)
        else:
            from src.preprocess.stemmer import get_stemmer
            analyzer = default_analyzer(get_stemmer(args.bm25_stemmer))
        print(f">>> BM25 (стеммер={args.bm25_stemmer})")
        bm = BM25Index(analyzer=analyzer).index(corpus_pairs)
        run_bm25 = bm.run(qmap_bm, top_k=DEPTH)

        def rrf(runs, k, top_k=10):
            fused = {}
            for qid in set().union(*[set(r) for r in runs]):
                sc = {}
                for r in runs:
                    for rank, doc in enumerate(r.get(qid, []), start=1):
                        sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
                fused[qid] = [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]
            return fused
        run_hyb = rrf([run_ft, run_bm25], k=args.rrf_k, top_k=10)

    categories = sorted(set(cats.values()))
    rows = []

    def block(name, sub_qrels):
        base = metrics.evaluate_run(run_base, sub_qrels, metrics=("ndcg", "mrr", "recall"), ks=(10,))
        ft = metrics.evaluate_run(run_ft, sub_qrels, metrics=("ndcg", "mrr", "recall"), ks=(10,))
        delta, p, _ = metrics.paired_bootstrap(run_base, run_ft, sub_qrels,
                                               metric="ndcg", k=10, n_resamples=args.n_resamples)
        row = {
            "scope": name,
            "ndcg@10_zeroshot": round(base["ndcg@10"], 4),
            "ndcg@10_finetuned": round(ft["ndcg@10"], 4),
            "delta": round(delta, 4),
            "p_value": round(p, 4),
            "mrr@10_zeroshot": round(base["mrr@10"], 4),
            "mrr@10_finetuned": round(ft["mrr@10"], 4),
        }
        if args.hybrid:
            row["ndcg@10_bm25"] = round(
                metrics.evaluate_run(run_bm25, sub_qrels, metrics=("ndcg",), ks=(10,))["ndcg@10"], 4)
            row["ndcg@10_hybrid"] = round(
                metrics.evaluate_run(run_hyb, sub_qrels, metrics=("ndcg",), ks=(10,))["ndcg@10"], 4)
        rows.append(row)

    block("ALL", qrels)
    for c in categories:
        qids = {qid for qid, cc in cats.items() if cc == c}
        block(c, {qi: rel for qi, rel in qrels.items() if qi in qids})

    print("\n=== nDCG@10 ===")
    if args.hybrid:
        print(f"{'scope':<16}{'zero-shot':>10}{'fine-tuned':>12}{'BM25':>9}{'FT⊕BM25':>10}{'p':>8}")
        for r in rows:
            sig = "*" if r["p_value"] < 0.05 else " "
            print(f"{r['scope']:<16}{r['ndcg@10_zeroshot']:>10.3f}{r['ndcg@10_finetuned']:>12.3f}"
                  f"{r['ndcg@10_bm25']:>9.3f}{r['ndcg@10_hybrid']:>10.3f}{r['p_value']:>8.3f}{sig}")
        print(f"  * = ft>zs значим (p<0.05). BM25-стеммер: {args.bm25_stemmer}. Δ = ft − zs.")
    else:
        print(f"{'scope':<16}{'zero-shot':>10}{'fine-tuned':>12}{'Δ':>9}{'p':>8}")
        for r in rows:
            sig = "*" if r["p_value"] < 0.05 else " "
            print(f"{r['scope']:<16}{r['ndcg@10_zeroshot']:>10.3f}"
                  f"{r['ndcg@10_finetuned']:>12.3f}{r['delta']:>+9.3f}{r['p_value']:>8.3f}{sig}")
        print("  * = значимо (p<0.05, paired bootstrap 10k). Δ = fine-tuned − zero-shot.")

    vs_rows = []
    if run_vs is not None:
        print(f"\n=== Дообученная vs {args.vs_model} (nDCG@10, paired bootstrap) ===")
        print(f"{'scope':<16}{'fine-tuned':>12}{'vs-model':>12}{'Δ':>9}{'p':>8}")
        for name, sub in ([("ALL", qrels)] +
                          [(c, {qi: rel for qi, rel in qrels.items()
                                if qi in {q for q, cc in cats.items() if cc == c}})
                           for c in categories]):
            ft_n = metrics.evaluate_run(run_ft, sub, metrics=("ndcg",), ks=(10,))["ndcg@10"]
            vs_n = metrics.evaluate_run(run_vs, sub, metrics=("ndcg",), ks=(10,))["ndcg@10"]
            # paired_bootstrap(A, B) → Δ = B − A; берём Δ = fine-tuned − vs
            d, p, _ = metrics.paired_bootstrap(run_vs, run_ft, sub, metric="ndcg",
                                               k=10, n_resamples=args.n_resamples)
            sig = "*" if p < 0.05 else " "
            print(f"{name:<16}{ft_n:>12.3f}{vs_n:>12.3f}{d:>+9.3f}{p:>8.3f}{sig}")
            vs_rows.append({"scope": name, "finetuned": round(ft_n, 4),
                            "vs_model": round(vs_n, 4), "delta": round(d, 4),
                            "p_value": round(p, 4)})
        print(f"  * = разница значима (p<0.05). Δ = fine-tuned − {args.vs_model}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "base_model": args.base_model,
        "finetuned": args.finetuned,
        "vs_model": args.vs_model,
        "vs_rows": vs_rows,
        "n_resamples": args.n_resamples,
        "rows": rows,
        "reference_ndcg10_all": {
            "kazakh-e5 (dense)": 0.747, "multilingual-e5-base": 0.785,
            "kazakh-e5 ⊕ BM25 (hybrid)": 0.808, "granite-107m zero-shot": 0.617,
        },
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nСохранено → {out}")
    print("\nГибрид ⊕BM25 (сильнейшая связка на этом бенчмарке):")
    print("  1) сохрани ранжировки FT-модели: "
          "python -m src.eval.run_dense --model <path> --top-k 100 --runs-out results/runs_ft.json")
    print("  2) слей с BM25 через штатный RRF: python -m src.eval.run_hybrid (см. его --help)")


if __name__ == "__main__":
    main()
