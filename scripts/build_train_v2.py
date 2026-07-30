#!/usr/bin/env python3
"""
Сборка обучающего файла v2 (полные тексты) из компактных артефактов.

Зачем: файл с полными текстами позитивов+негативов ~240 МБ (GitHub не держит
>100 МБ). Поэтому в репо лежат лёгкие артефакты, а тяжёлый train-файл собирается
на месте (Kaggle/Colab) этим скриптом:

  • --ids   data/synthetic_pairs.hn.ids.jsonl  (query + positive_id + negative_ids)
  • --gold  data/kazqad_pairs.dedup.jsonl      (KazQAD gold, уже с текстами)
  • --corpus '.../corpus/*.jsonl.gz'           (KazQAD, резолв id→текст)

Выход — JSONL в формате train.py: {query, positive, negatives:[...], category}.

Пример:
    python scripts/build_train_v2.py \\
        --ids data/synthetic_pairs.hn.ids.jsonl \\
        --gold data/kazqad_pairs.dedup.jsonl \\
        --corpus '/content/KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \\
        --out data/train_pairs_v2.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import load_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка train-файла v2 (id → тексты).")
    ap.add_argument("--ids", required=True, help="synthetic_pairs.hn.ids.jsonl (query/positive_id/negative_ids).")
    ap.add_argument("--gold", default=None, help="KazQAD gold с текстами (kazqad_pairs.dedup.jsonl).")
    ap.add_argument("--corpus", required=True, help="Glob к пассажам KazQAD (.jsonl[.gz]).")
    ap.add_argument("--out", default="data/train_pairs_v2.jsonl")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.expanduser(args.corpus)))
    if not files:
        raise SystemExit(f"Корпус не найден: {args.corpus}")
    print(f"Загрузка корпуса ({len(files)} файл(ов)) …")
    corpus = load_corpus([Path(f) for f in files])
    print(f"  пассажей: {len(corpus):,}")

    def txt(did):
        d = corpus.get(did)
        return d["text"] if d else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_syn = n_neg = miss_pos = miss_neg = 0
    with open(args.ids, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pos = txt(d.get("positive_id", ""))
            if not pos:
                miss_pos += 1
                continue
            negs = []
            for nid in d.get("negative_ids", []):
                t = txt(nid)
                if t:
                    negs.append(t)
                else:
                    miss_neg += 1
            o.write(json.dumps({"query": d["query"], "positive": pos,
                                "negatives": negs, "category": d.get("category")},
                               ensure_ascii=False) + "\n")
            n_syn += 1
            n_neg += len(negs)

        n_gold = 0
        if args.gold and os.path.exists(args.gold):
            with open(args.gold, encoding="utf-8") as g:
                for line in g:
                    if line.strip():
                        o.write(line if line.endswith("\n") else line + "\n")
                        n_gold += 1

    print(f"Синтетика: {n_syn:,} (негативов {n_neg:,}) | пропущено pos {miss_pos}, neg {miss_neg}")
    if args.gold:
        print(f"KazQAD gold: {n_gold:,}")
    print(f"Итого строк: {n_syn + n_gold:,} → {out}")


if __name__ == "__main__":
    main()
