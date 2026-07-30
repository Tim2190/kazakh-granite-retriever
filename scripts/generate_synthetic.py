#!/usr/bin/env python3
"""
Синтетическая генерация обучающих пар (query → passage) для казахского retrieval.

Зачем: KazQAD даёт лишь ~3.9K размеченных позитивов → мало, файнтюн переобучается
(natural проседает). Генерируем десятки тысяч пар на пассажах корпуса казвики, покрывая
ТРИ навыка сразу (как в бенчмарке, но на непересекающихся статьях):
  • natural     — прямой фактический вопрос (чтобы модель НЕ забыла natural);
  • vocab_gap   — парафраз/синонимы, без лексического пересечения с пассажем;
  • inflected   — вопрос с падежами/морфологией.

Антилик: на вход подаём пул пассажей, УЖЕ очищенный от статей бенчмарка и второго
(ручного) тестового датасета — см. --exclude-article-ids. Плюс генерация идёт по
статьям, которых нет в исключениях.

Провайдеры (ключ из окружения):
  • claude  → ANTHROPIC_API_KEY   (по умолч. модель claude-haiku-4-5-20251001)
  • gemini  → GEMINI_API_KEY      (по умолч. модель gemini-2.0-flash)

Выход — JSONL: {"query","positive","category","passage_id","positive_title"}.
Резюмируемо: повторный запуск дописывает, пропуская уже обработанные passage_id.

Пример:
    export GEMINI_API_KEY=...
    python scripts/generate_synthetic.py \\
        --corpus '/content/KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \\
        --provider gemini --n-passages 14000 --batch 5 \\
        --exclude-article-ids data/exclude_article_ids.txt \\
        --out data/synthetic_pairs.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# переиспользуем загрузчик корпуса и вытяжку article_id из doc_id
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import load_corpus, article_id  # noqa: E402

PROMPT = """You are building a Kazakh information-retrieval training set.
For EACH numbered passage below, write THREE Kazakh questions that are answerable \
ONLY from that passage. Return STRICT JSON: a list of objects, one per passage, \
with fields: "i" (the passage number), "natural", "vocab_gap", "inflected".

Rules for each question type (all in fluent, natural Kazakh):
- "natural": a direct factual question about a key fact in the passage.
- "vocab_gap": a paraphrased question that asks the SAME thing using SYNONYMS and \
different wording — it MUST NOT reuse the salient content words from the passage \
(test semantic, not lexical, matching).
- "inflected": a question that uses grammatical cases / morphological variation of \
the key entities (oblique cases), still natural Kazakh.

Constraints: no yes/no questions; each question must be self-contained and clearly \
answerable from its passage alone; do not include the answer in the question; \
output ONLY the JSON list, nothing else.

Passages:
{passages}
"""


# ------------------------------- LLM провайдеры -------------------------------

def _post(url: str, headers: Dict[str, str], body: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call_claude(prompt: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Нет ANTHROPIC_API_KEY в окружении.")
    resp = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": model, "max_tokens": 2000,
         "messages": [{"role": "user", "content": prompt}]},
    )
    return resp["content"][0]["text"]


def call_gemini(prompt: str, model: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Нет GEMINI_API_KEY в окружении.")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    resp = _post(url, {"content-type": "application/json"},
                 {"contents": [{"parts": [{"text": prompt}]}]})
    return resp["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {"claude": call_claude, "gemini": call_gemini}
DEFAULT_MODEL = {"claude": "claude-haiku-4-5-20251001", "gemini": "gemini-2.0-flash"}


# --------------------------------- парсинг ------------------------------------

def extract_json_list(text: str) -> Optional[list]:
    """Достаём JSON-массив из ответа (сносим ```-обёртки и мусор по краям)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------- утилиты -----------------------------------

def load_exclusions(path: Optional[str]) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def done_passage_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["passage_id"])
            except Exception:
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Синтетические (query, passage) пары для казахского retrieval")
    ap.add_argument("--corpus", required=True, help="Путь/glob к пассажам казвики (KazQAD corpus .jsonl[.gz]).")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ap.add_argument("--model", default=None, help="Переопределить модель провайдера.")
    ap.add_argument("--n-passages", type=int, default=14000,
                    help="Сколько пассажей обработать (×3 вопроса ≈ итоговые пары).")
    ap.add_argument("--batch", type=int, default=5, help="Пассажей на один запрос к LLM.")
    ap.add_argument("--min-chars", type=int, default=200, help="Мин. длина пассажа (отсев огрызков).")
    ap.add_argument("--exclude-article-ids", default=None,
                    help="Файл со списком article_id для исключения (бенчмарк + 2-й датасет).")
    ap.add_argument("--out", default="data/synthetic_pairs.jsonl")
    ap.add_argument("--sleep", type=float, default=0.5, help="Пауза между запросами (rate-limit).")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    model = args.model or DEFAULT_MODEL[args.provider]
    call = PROVIDERS[args.provider]
    random.seed(args.seed)

    files = sorted(glob.glob(os.path.expanduser(args.corpus)))
    if not files:
        raise SystemExit(f"Пассажи не найдены: {args.corpus}")
    print(f"Загрузка корпуса: {len(files)} файл(ов) …")
    corpus = load_corpus([Path(f) for f in files])

    exclude = load_exclusions(args.exclude_article_ids)
    print(f"Исключений (article_id): {len(exclude):,}")

    # пул кандидатов: достаточной длины и не из исключённых статей
    pool = [(did, d["text"], d["title"]) for did, d in corpus.items()
            if len(d["text"]) >= args.min_chars and article_id(did) not in exclude]
    random.shuffle(pool)
    print(f"Пул пассажей после фильтра: {len(pool):,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    already = done_passage_ids(out)
    if already:
        print(f"Резюм: уже обработано {len(already):,} — пропускаю.")
    pool = [p for p in pool if p[0] not in already][: args.n_passages]
    print(f"К обработке: {len(pool):,} пассажей, провайдер={args.provider}, модель={model}\n")

    written = 0
    fout = open(out, "a", encoding="utf-8")
    for i in range(0, len(pool), args.batch):
        chunk = pool[i:i + args.batch]
        listing = "\n".join(f"[{j}] {text}" for j, (_, text, _) in enumerate(chunk))
        try:
            raw = call(PROMPT.format(passages=listing), model)
        except Exception as e:
            print(f"  [warn] запрос упал ({e}) — пауза и дальше")
            time.sleep(2.0)
            continue
        items = extract_json_list(raw) or []
        by_i = {it.get("i"): it for it in items if isinstance(it, dict)}
        for j, (did, text, title) in enumerate(chunk):
            it = by_i.get(j)
            if not it:
                continue
            for cat in ("natural", "vocab_gap", "inflected"):
                q = (it.get(cat) or "").strip()
                if len(q) < 5:
                    continue
                fout.write(json.dumps({
                    "query": q, "positive": text, "category": cat,
                    "passage_id": did, "positive_title": title,
                }, ensure_ascii=False) + "\n")
                written += 1
        fout.flush()
        if (i // args.batch) % 20 == 0:
            print(f"  …{i + len(chunk):,}/{len(pool):,} пассажей, пар записано: {written:,}")
        time.sleep(args.sleep)
    fout.close()
    print(f"\nГотово. Пар записано за прогон: {written:,} → {out}")
    print("Дальше: слить с KazQAD-парами → check_overlap.py (антилик) → train.py.")


if __name__ == "__main__":
    main()
