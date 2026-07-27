#!/usr/bin/env python3
"""
Шаг в — KazQAD → тренировочные тройки для sentence-transformers.

KazQAD (github.com/IS2AI/KazQAD, CC BY-SA) отдаёт IR-данные в TREC-layout:
    data/information-retrieval/
        corpus/   — 800K+ пассажей казахской Википедии (в git может отсутствовать!)
        topics/   — kazqad-topics-v1.0-kk-{train,validation,test}.tsv  →  qid \\t query
        qrels/    — kazqad-qrels-v1.0-{train,validation,test}.tsv      →  qid \\t 0 \\t docid \\t rel

Ключевые факты формата (подтверждены --inspect на v1.0):
  • doc_id = "<article_id>_<para>_<chunk>", напр. 493371_1_1 → статья Википедии 493371.
  • qrels содержат И rel>0 (позитивы), И rel=0 — это РАЗМЕЧЕННЫЕ hard-negatives:
    пассажи, попавшие в пул кандидатов и признанные нерелевантными. Тянем их в
    обучение как жёсткие негативы — прямо в мишень слабости Granite на vocab-gap.

Выход — JSONL, по строке на (query, positive) с прикреплённым списком негативов:
    {"query", "positive", "negatives":[...], "query_id",
     "positive_id", "negative_ids":[...], "positive_title", "split"}

Использование (Colab, после клона KazQAD):
    # инспекция формата
    python scripts/prepare_data.py --kazqad-root /content/KazQAD/data/information-retrieval --inspect
    # сборка (train; validation держим для мониторинга обучения, test — вне)
    python scripts/prepare_data.py \\
        --kazqad-root /content/KazQAD/data/information-retrieval \\
        --splits train \\
        --corpus-file /path/to/corpus.tsv[.gz] \\   # если corpus/ пуст в git
        --out data/kazqad_pairs.jsonl
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ID_KEYS = ("id", "_id", "doc_id", "docid", "pid", "passage_id")
_TITLE_KEYS = ("title", "article_title", "wiki_title", "heading")
_TEXT_KEYS = ("text", "contents", "passage", "body", "content", "document")
_QTEXT_KEYS = ("text", "query", "question", "title")
_QID_KEYS = ("id", "_id", "qid", "query_id", "question_id")


def article_id(doc_id: str) -> str:
    """'493371_1_1' -> '493371' (id статьи Википедии). Ключ для дедупа."""
    return doc_id.split("_", 1)[0]


# ----------------------------- обнаружение файлов -----------------------------

def _find_dir(root: Path, name: str) -> Optional[Path]:
    p = root / name
    return p if p.is_dir() else None


def _data_files(d: Optional[Path]) -> List[Path]:
    if d is None:
        return []
    out: List[Path] = []
    for pat in ("*.jsonl", "*.json", "*.tsv", "*.txt", "*.csv",
                "*.jsonl.gz", "*.tsv.gz", "*.csv.gz"):
        out += [Path(p) for p in glob.glob(str(d / "**" / pat), recursive=True)]
    return sorted(set(out))


def _match_split(path: Path, split: str) -> bool:
    return split.lower() in path.name.lower()


# ------------------------------- загрузка данных ------------------------------

def _first_present(obj: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return str(obj[k])
    return None


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _iter_lines(path: Path) -> Iterable[str]:
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def _is_json(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".jsonl") or name.endswith(".json") or name.endswith(".jsonl.gz")


def load_corpus(path: Path) -> Dict[str, Dict[str, str]]:
    """doc_id -> {"title":..., "text":...}. JSONL или TSV(id[,title],text), опц. .gz."""
    corpus: Dict[str, Dict[str, str]] = {}
    is_json = _is_json(path)
    for line in _iter_lines(path):
        if is_json:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = _first_present(obj, _ID_KEYS)
            if did is None:
                continue
            corpus[did] = {
                "title": _first_present(obj, _TITLE_KEYS) or "",
                "text": _first_present(obj, _TEXT_KEYS) or "",
            }
        else:
            parts = line.split("\t")
            if len(parts) == 2:
                did, title, text = parts[0], "", parts[1]
            elif len(parts) >= 3:
                did, title, text = parts[0], parts[1], parts[2]
            else:
                continue
            corpus[did] = {"title": title, "text": text}
    return corpus


def load_topics(path: Path) -> Dict[str, str]:
    """qid -> текст запроса. TSV(qid, query) или JSONL."""
    topics: Dict[str, str] = {}
    is_json = _is_json(path)
    for line in _iter_lines(path):
        if is_json:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = _first_present(obj, _QID_KEYS)
            qtext = _first_present(obj, _QTEXT_KEYS)
            if qid and qtext:
                topics[qid] = qtext
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                topics[parts[0]] = parts[1]
    return topics


def load_qrels(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    qid -> {"pos":[doc_id...], "neg":[doc_id...]}.
    TREC(qid 0 docid rel) или TSV(qid docid rel). rel>0 → pos, rel==0 → hard-neg.
    """
    qrels: Dict[str, Dict[str, List[str]]] = {}
    for line in _iter_lines(path):
        parts = line.split()
        if len(parts) == 4:            # qid 0 docid rel
            qid, _, did, rel = parts
        elif len(parts) == 3:          # qid docid rel
            qid, did, rel = parts
        elif len(parts) == 2:          # qid docid (rel=1)
            qid, did, rel = parts[0], parts[1], "1"
        else:
            continue
        try:
            r = float(rel)
        except ValueError:
            continue
        bucket = qrels.setdefault(qid, {"pos": [], "neg": []})
        bucket["pos" if r > 0 else "neg"].append(did)
    return qrels


# ---------------------------------- inspect -----------------------------------

def inspect(root: Path) -> None:
    print(f"KazQAD IR root: {root}\n")
    for sub in ("corpus", "topics", "qrels"):
        d = _find_dir(root, sub)
        files = _data_files(d)
        print(f"[{sub}] {'(нет папки)' if d is None else d}")
        if not files:
            print("   — файлов не найдено\n")
            continue
        for fp in files:
            print(f"   {fp.name}  ({fp.stat().st_size/1e6:.1f} MB)")
            for i, line in enumerate(_iter_lines(fp)):
                print(f"      | {line[:200]}")
                if i >= 1:
                    break
        print()


# ----------------------------------- build ------------------------------------

def _resolve_corpus(root: Path, corpus_file: Optional[str]) -> Path:
    if corpus_file:
        p = Path(corpus_file).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--corpus-file не найден: {p}")
        return p
    files = _data_files(_find_dir(root, "corpus"))
    if not files:
        raise SystemExit(
            "Корпус KazQAD не найден в corpus/ (в git его часто нет).\n"
            "Найди файл пассажей (см. диагностическую ячейку) и передай --corpus-file <путь>."
        )
    return files[0]


def build(root: Path, splits: List[str], corpus_file: Optional[str],
          max_neg: int) -> Tuple[List[dict], dict]:
    cpath = _resolve_corpus(root, corpus_file)
    print(f"Загрузка корпуса: {cpath.name} …")
    corpus = load_corpus(cpath)
    print(f"  пассажей: {len(corpus):,}")
    has_titles = any(v["title"] for v in list(corpus.values())[:1000])
    print(f"  заголовки статей в корпусе: {'есть' if has_titles else 'НЕТ (дедуп — по article_id/тексту)'}")

    topics_files = _data_files(_find_dir(root, "topics"))
    qrels_files = _data_files(_find_dir(root, "qrels"))

    pairs: List[dict] = []
    stats = {"splits": {}, "missing_pos_doc": 0, "missing_neg_doc": 0,
             "missing_query": 0, "neg_total": 0}
    for split in splits:
        tf = next((f for f in topics_files if _match_split(f, split)), None)
        qf = next((f for f in qrels_files if _match_split(f, split)), None)
        if tf is None or qf is None:
            print(f"  [пропуск] split={split}: topics={tf}, qrels={qf}")
            continue
        topics = load_topics(tf)
        qrels = load_qrels(qf)
        n0 = len(pairs)
        for qid, rel in qrels.items():
            qtext = topics.get(qid)
            if not qtext:
                stats["missing_query"] += 1
                continue
            neg_texts, neg_ids = [], []
            for nd in rel["neg"][:max_neg]:
                doc = corpus.get(nd)
                if doc and doc["text"]:
                    neg_texts.append(doc["text"])
                    neg_ids.append(nd)
                else:
                    stats["missing_neg_doc"] += 1
            for pd in rel["pos"]:
                doc = corpus.get(pd)
                if doc is None or not doc["text"]:
                    stats["missing_pos_doc"] += 1
                    continue
                pairs.append({
                    "query": qtext,
                    "positive": doc["text"],
                    "negatives": neg_texts,
                    "query_id": qid,
                    "positive_id": pd,
                    "negative_ids": neg_ids,
                    "positive_title": doc["title"],
                    "split": split,
                })
                stats["neg_total"] += len(neg_texts)
        stats["splits"][split] = len(pairs) - n0
        print(f"  split={split}: пар +{len(pairs)-n0} "
              f"(topics={len(topics)}, qrels-запросов={len(qrels)})")
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="KazQAD → тройки (query, positive, hard-negatives)")
    ap.add_argument("--kazqad-root", required=True,
                    help="Путь к data/information-retrieval клонированного KazQAD.")
    ap.add_argument("--inspect", action="store_true",
                    help="Только показать найденные файлы и первые строки.")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="Сплиты KazQAD: train | validation | test. По умолч.: train.")
    ap.add_argument("--corpus-file", default=None,
                    help="Явный путь к файлу корпуса (если corpus/ пуст). Поддержка .gz.")
    ap.add_argument("--max-neg", type=int, default=8,
                    help="Сколько hard-negatives (rel=0) тянуть на запрос. По умолч.: 8.")
    ap.add_argument("--out", default="data/kazqad_pairs.jsonl")
    args = ap.parse_args()

    root = Path(args.kazqad_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Нет такого пути: {root}")

    if args.inspect:
        inspect(root)
        return

    pairs, stats = build(root, args.splits, args.corpus_file, args.max_neg)
    if not pairs:
        raise SystemExit("0 пар. Проверь --corpus-file / --splits и вывод --inspect.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_with_neg = sum(1 for p in pairs if p["negatives"])
    print(f"\nВсего пар (query, positive): {len(pairs):,}")
    print(f"  из них с hard-negatives: {n_with_neg:,}; негативов всего: {stats['neg_total']:,}")
    print(f"  по сплитам: {stats['splits']}")
    print(f"  пропущено: pos-doc нет в корпусе {stats['missing_pos_doc']:,}, "
          f"neg-doc {stats['missing_neg_doc']:,}, текста запроса {stats['missing_query']:,}")
    print(f"Сохранено → {out}")
    print("\nДальше: check_overlap.py уберёт тройки со статьями из бенчмарка (антилик), потом train.py.")


if __name__ == "__main__":
    main()
