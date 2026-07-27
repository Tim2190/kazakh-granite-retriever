#!/usr/bin/env python3
"""
Шаг г — дедупликация: убрать из тренировочного KazQAD всё, что пересекается с
оценочным бенчмарком (Kaz-RAG-search-benchmark), ДО обучения.

Зачем это критично: оба корпуса восходят к казахской Википедии
(бенчмарк — 800 случайных статей из wikimedia/wikipedia; KazQAD — 800K+ пассажей
казвики). Без дедупа тренировка увидит оценочные пассажи → train/test leakage →
завышенные метрики и мёртвый проект. Это единственный шаг, который может обнулить
доверие к результату, поэтому чистим агрессивно и с отчётом.

Три слоя обнаружения пересечений (id-пространства сторон РАЗНЫЕ:
бенчмарк 'wiki_4009' — индекс HF-датасета; KazQAD '493371_1_1' — pageid казвики,
поэтому матч по article_id сам по себе ненадёжен):
  1. ЗАГОЛОВОК статьи   — id-независимый матч (если у бенчмарк-корпуса есть title).
  2. NEAR-DUP по тексту — инвертированный индекс шинглов + точный Jaccard; ловит
     тот же текст при любом чанкинге. Самый надёжный слой.
  3. ARTICLE_ID         — по умолчанию только ДИАГНОСТИКА (report покажет, совпали
     ли id-пространства). Включить дроп по id: --drop-by-article-id.

Чистятся и позитивы (дроп всей тройки), и негативы (gold-пассаж бенчмарка не должен
попасть в hard-negatives — иначе учим модель опускать оценочный ответ).

Вход:  --pairs (выход prepare_data.py), --benchmark-corpus (corpus.jsonl бенчмарка).
Выход: --out (очищенные тройки) + --report (JSON, также печатается).

Пример:
    python scripts/check_overlap.py \\
        --pairs data/kazqad_pairs.jsonl \\
        --benchmark-corpus /content/Kaz-RAG-search-benchmark/data/corpus/corpus.jsonl \\
        --out data/kazqad_pairs.dedup.jsonl \\
        --report results/overlap_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ID_KEYS = ("id", "_id", "doc_id", "docid", "pid", "passage_id")
_TITLE_KEYS = ("title", "article_title", "wiki_title", "heading")
_TEXT_KEYS = ("text", "contents", "passage", "body", "content", "document")

_WORD = re.compile(r"\w+", re.UNICODE)


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def shingles(toks: List[str], n: int) -> Set[int]:
    """Множество хэшей словных n-грамм (экономим память на длинных текстах)."""
    if len(toks) < n:
        return {hash(" ".join(toks))} if toks else set()
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


# ----------------------------- бенчмарк-сигнатуры -----------------------------

class BenchmarkIndex:
    """Заголовки + article_id + near-dup индекс по пассажам бенчмарка."""

    def __init__(self, n: int, probe: int):
        self.n = n
        self.probe = probe                 # сколько шинглов запроса зондировать (LSH-lite)
        self.titles: Set[str] = set()
        self.article_ids: Set[str] = set()
        self.shingle_sets: List[Set[int]] = []
        self.inverted: Dict[int, List[int]] = {}
        self.has_titles = False

    def add(self, doc_id: str, title: str, text: str, id_regex: re.Pattern) -> None:
        if title:
            self.titles.add(norm_title(title))
            self.has_titles = True
        m = id_regex.search(doc_id or "")
        if m:
            self.article_ids.add(m.group(1))
        sh = shingles(tokens(text), self.n)
        idx = len(self.shingle_sets)
        self.shingle_sets.append(sh)
        for s in sorted(sh)[: self.probe]:      # индексируем зондирующее подмножество
            self.inverted.setdefault(s, []).append(idx)

    def near_dup(self, text: str, thr: float) -> Tuple[bool, float]:
        q = shingles(tokens(text), self.n)
        if not q:
            return False, 0.0
        cand: Set[int] = set()
        for s in sorted(q)[: self.probe]:
            cand.update(self.inverted.get(s, ()))
        best = 0.0
        for i in cand:
            b = self.shingle_sets[i]
            inter = len(q & b)
            if not inter:
                continue
            j = inter / len(q | b)
            if j > best:
                best = j
                if best >= thr:
                    return True, best
        return best >= thr, best


def load_benchmark(path: Path, n: int, probe: int, id_regex: re.Pattern) -> BenchmarkIndex:
    bi = BenchmarkIndex(n, probe)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = next((str(obj[k]) for k in _ID_KEYS if k in obj), "")
            title = next((str(obj[k]) for k in _TITLE_KEYS if obj.get(k)), "")
            text = next((str(obj[k]) for k in _TEXT_KEYS if obj.get(k)), "")
            bi.add(did, title, text, id_regex)
    return bi


# ----------------------------------- дедуп ------------------------------------

def art_id(doc_id: str) -> str:
    return (doc_id or "").split("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Дедуп KazQAD против оценочного бенчмарка")
    ap.add_argument("--pairs", required=True, help="JSONL из prepare_data.py")
    ap.add_argument("--benchmark-corpus", required=True, help="corpus.jsonl бенчмарка")
    ap.add_argument("--out", default="data/kazqad_pairs.dedup.jsonl")
    ap.add_argument("--report", default="results/overlap_report.json")
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--probe", type=int, default=48,
                    help="Сколько шинглов зондировать при near-dup (скорость/полнота).")
    ap.add_argument("--jaccard", type=float, default=0.5,
                    help="Порог near-dup Jaccard для признания пересечением.")
    ap.add_argument("--benchmark-id-regex", default=r"wiki_(\d+)_",
                    help="Regex для article_id из doc_id бенчмарка (group 1).")
    ap.add_argument("--drop-by-article-id", action="store_true",
                    help="Также дропать по совпадению article_id (по умолч. только диагностика).")
    ap.add_argument("--skip-neg-check", action="store_true",
                    help="Не чистить негативы near-dup'ом (быстрее).")
    args = ap.parse_args()

    id_regex = re.compile(args.benchmark_id_regex)
    print(f"Индексирую бенчмарк: {args.benchmark_corpus}")
    bi = load_benchmark(Path(args.benchmark_corpus), args.shingle_size, args.probe, id_regex)
    print(f"  пассажей: {len(bi.shingle_sets):,} | статей(article_id): {len(bi.article_ids):,} | "
          f"заголовки: {'есть' if bi.has_titles else 'НЕТ'}")
    if not bi.has_titles:
        print("  [i] У бенчмарк-корпуса нет title → слой заголовков отключён, "
              "опираемся на near-dup (+ article_id при --drop-by-article-id).")

    reasons = {"title": 0, "near_dup": 0, "article_id": 0}
    art_id_overlap_diag = 0          # диагностика совпадения id-пространств
    kept: List[dict] = []
    neg_stripped = 0
    total = 0

    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            p = json.loads(line)
            pid = p.get("positive_id", "")
            a = art_id(pid)
            id_hit = a in bi.article_ids
            if id_hit:
                art_id_overlap_diag += 1

            leak_reason = None
            if bi.has_titles and norm_title(p.get("positive_title", "")) in bi.titles:
                leak_reason = "title"
            else:
                dup, _ = bi.near_dup(p.get("positive", ""), args.jaccard)
                if dup:
                    leak_reason = "near_dup"
                elif args.drop_by_article_id and id_hit:
                    leak_reason = "article_id"

            if leak_reason:
                reasons[leak_reason] += 1
                continue

            # позитив чист — почистим негативы от пассажей бенчмарка
            negs, neg_ids = p.get("negatives", []), p.get("negative_ids", [])
            if negs:
                keep_n, keep_id = [], []
                for txt, nid in zip(negs, neg_ids):
                    bad = (args.drop_by_article_id and art_id(nid) in bi.article_ids)
                    if not bad and not args.skip_neg_check:
                        bad, _ = bi.near_dup(txt, args.jaccard)
                    if bad:
                        neg_stripped += 1
                    else:
                        keep_n.append(txt)
                        keep_id.append(nid)
                p["negatives"], p["negative_ids"] = keep_n, keep_id
            kept.append(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    dropped = total - len(kept)
    report = {
        "pairs_in": total,
        "pairs_kept": len(kept),
        "pairs_dropped": dropped,
        "drop_reasons": reasons,
        "negatives_stripped": neg_stripped,
        "article_id_overlap_diagnostic": art_id_overlap_diag,
        "benchmark_passages": len(bi.shingle_sets),
        "benchmark_articles": len(bi.article_ids),
        "params": {
            "shingle_size": args.shingle_size, "jaccard": args.jaccard,
            "probe": args.probe, "drop_by_article_id": args.drop_by_article_id,
            "benchmark_has_titles": bi.has_titles,
        },
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(rep, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== Отчёт дедупликации ===")
    print(f"  вход:      {total:,} троек")
    print(f"  выкинуто:  {dropped:,}  (title={reasons['title']:,}, "
          f"near_dup={reasons['near_dup']:,}, article_id={reasons['article_id']:,})")
    print(f"  осталось:  {len(kept):,}")
    print(f"  негативов вычищено: {neg_stripped:,}")
    print(f"  [диагностика] совпадений article_id с бенчмарком: {art_id_overlap_diag:,}")
    if art_id_overlap_diag and not args.drop_by_article_id:
        print("     ↳ id-пространства, похоже, пересекаются — рассмотри --drop-by-article-id "
              "для более строгой чистки по целым статьям.")
    print(f"\nОчищено → {out}\nОтчёт → {rep}")


if __name__ == "__main__":
    main()
