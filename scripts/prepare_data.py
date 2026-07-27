#!/usr/bin/env python3
"""
Шаг в — KazQAD → тренировочные пары для sentence-transformers.

KazQAD (github.com/IS2AI/KazQAD, CC BY-SA) отдаёт IR-данные в TREC-layout:
    data/information-retrieval/
        corpus/   — 800K+ пассажей казахской Википедии
        topics/   — вопросы (train/dev/test)
        qrels/    — суждения релевантности (qid → doc_id)

Скрипт джойнит topics+qrels+corpus и пишет JSONL пар (query, positive) под
MultipleNegativesRankingLoss. Hard-negatives добавим отдельным шагом (майнинг
BM25/зеро-шот Granite) — для MNRL достаточно позитивов + in-batch негативов,
но поля под негативы заложены.

ВАЖНО: точные имена файлов и поля KazQAD я по исходнику не проверял (GitHub API
отдаёт только имена папок), поэтому парсер авто-детектит формат. Прогони СНАЧАЛА
режим --inspect, чтобы увидеть реальные файлы и первые строки — если авто-детект
что-то не так разберёт, поправим маппинг полей точечно.

Использование (в Colab, после клона KazQAD):
    # 1) посмотреть, что реально лежит
    python scripts/prepare_data.py \
        --kazqad-root /content/KazQAD/data/information-retrieval --inspect

    # 2) собрать пары
    python scripts/prepare_data.py \
        --kazqad-root /content/KazQAD/data/information-retrieval \
        --splits train dev \
        --out data/kazqad_pairs.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Возможные имена ключей в разных дампах — авто-детект берёт первый попавшийся.
_ID_KEYS = ("id", "_id", "doc_id", "docid", "pid", "passage_id")
_TITLE_KEYS = ("title", "article_title", "wiki_title", "heading")
_TEXT_KEYS = ("text", "contents", "passage", "body", "content", "document")
_QTEXT_KEYS = ("text", "query", "question", "title")
_QID_KEYS = ("id", "_id", "qid", "query_id", "question_id")


# ----------------------------- обнаружение файлов -----------------------------

def _find_dir(root: Path, name: str) -> Optional[Path]:
    p = root / name
    return p if p.is_dir() else None


def _data_files(d: Optional[Path]) -> List[Path]:
    if d is None:
        return []
    out: List[Path] = []
    for pat in ("*.jsonl", "*.json", "*.tsv", "*.txt", "*.csv"):
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


def _iter_lines(path: Path) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def load_corpus(path: Path) -> Dict[str, Dict[str, str]]:
    """doc_id -> {"title":..., "text":...}. Понимает JSONL и TSV(id[,title],text)."""
    corpus: Dict[str, Dict[str, str]] = {}
    is_json = path.suffix.lower() in (".jsonl", ".json")
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
                did, text, title = parts[0], parts[1], ""
            elif len(parts) >= 3:
                did, title, text = parts[0], parts[1], parts[2]
            else:
                continue
            corpus[did] = {"title": title, "text": text}
    return corpus


def load_topics(path: Path) -> Dict[str, str]:
    """qid -> текст запроса. Понимает JSONL и TSV(qid, query)."""
    topics: Dict[str, str] = {}
    is_json = path.suffix.lower() in (".jsonl", ".json")
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


def load_qrels(path: Path) -> Dict[str, List[str]]:
    """qid -> [doc_id, ...] с rel>0. Понимает TREC(qid 0 docid rel) и TSV(qid docid[ rel])."""
    qrels: Dict[str, List[str]] = {}
    for line in _iter_lines(path):
        parts = line.split()
        if len(parts) == 4:            # TREC: qid 0 docid rel
            qid, _, did, rel = parts
        elif len(parts) == 3:          # qid docid rel
            qid, did, rel = parts
        elif len(parts) == 2:          # qid docid  (rel подразумевается 1)
            qid, did, rel = parts[0], parts[1], "1"
        else:
            continue
        try:
            if float(rel) <= 0:
                continue
        except ValueError:
            continue
        qrels.setdefault(qid, []).append(did)
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
            size = fp.stat().st_size
            print(f"   {fp.name}  ({size/1e6:.1f} MB)")
            for i, line in enumerate(_iter_lines(fp)):
                print(f"      | {line[:200]}")
                if i >= 1:
                    break
        print()
    print("Если поля/разделители распознаны неверно — скинь этот вывод, поправлю маппинг.")


# ----------------------------------- build ------------------------------------

def build_pairs(root: Path, splits: List[str]) -> Tuple[List[dict], dict]:
    corpus_files = _data_files(_find_dir(root, "corpus"))
    topics_files = _data_files(_find_dir(root, "topics"))
    qrels_files = _data_files(_find_dir(root, "qrels"))
    if not corpus_files:
        raise SystemExit("Не найден corpus. Проверь --kazqad-root и запусти --inspect.")

    print(f"Загрузка корпуса: {corpus_files[0].name} …")
    corpus = load_corpus(corpus_files[0])
    print(f"  пассажей: {len(corpus):,}")

    pairs: List[dict] = []
    stats = {"splits": {}, "missing_docs": 0, "missing_query": 0}
    for split in splits:
        tf = next((f for f in topics_files if _match_split(f, split)), None)
        qf = next((f for f in qrels_files if _match_split(f, split)), None)
        if tf is None or qf is None:
            print(f"  [пропуск] split={split}: topics={tf}, qrels={qf}")
            continue
        topics = load_topics(tf)
        qrels = load_qrels(qf)
        n_before = len(pairs)
        for qid, doc_ids in qrels.items():
            qtext = topics.get(qid)
            if not qtext:
                stats["missing_query"] += 1
                continue
            for did in doc_ids:
                doc = corpus.get(did)
                if doc is None or not doc["text"]:
                    stats["missing_docs"] += 1
                    continue
                pairs.append({
                    "query": qtext,
                    "positive": doc["text"],
                    "query_id": qid,
                    "positive_id": did,          # для дедупа (check_overlap.py)
                    "positive_title": doc["title"],
                    "split": split,
                })
        stats["splits"][split] = len(pairs) - n_before
        print(f"  split={split}: пар +{len(pairs) - n_before} "
              f"(topics={len(topics)}, qrels-запросов={len(qrels)})")
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="KazQAD → пары (query, positive) под sentence-transformers")
    ap.add_argument("--kazqad-root", required=True,
                    help="Путь к data/information-retrieval клонированного KazQAD.")
    ap.add_argument("--inspect", action="store_true",
                    help="Только показать найденные файлы и первые строки (формат-детект).")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="Какие сплиты собирать (test держим вне обучения). По умолч.: train.")
    ap.add_argument("--out", default="data/kazqad_pairs.jsonl")
    args = ap.parse_args()

    root = Path(args.kazqad_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Нет такого пути: {root}")

    if args.inspect:
        inspect(root)
        return

    pairs, stats = build_pairs(root, args.splits)
    if not pairs:
        raise SystemExit("0 пар. Запусти --inspect и пришли вывод — поправлю парсер под форматы.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nВсего пар: {len(pairs):,}")
    print(f"По сплитам: {stats['splits']}")
    print(f"Пропущено (нет doc в корпусе): {stats['missing_docs']:,}; "
          f"(нет текста запроса): {stats['missing_query']:,}")
    print(f"Сохранено → {out}")
    print("\nДальше: check_overlap.py уберёт тройки, чьи статьи есть в бенчмарке "
          "(антилик), затем train.py.")


if __name__ == "__main__":
    main()
