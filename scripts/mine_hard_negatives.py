#!/usr/bin/env python3
"""
Майнинг hard-negatives для тренировочных пар (BM25 + казахский стеммер).

Идея: для каждого запроса находим пассажи с высоким лексическим пересечением
(общие корни после стемминга), но НЕ являющиеся золотым ответом — «обманки».
Обучение против них учит модель, что общие слова ≠ релевантность → бьёт в
vocab-gap. BM25 здесь — только инструмент подготовки данных; в модели его нет.

Стеммер — ТВОЙ (Kaz-RAG-search-benchmark/src/preprocess): get_stemmer("kazakh")
= Cloud Run сервис (батч 50, 30 req/min, дисковый кэш). Симметрично к запросу и
корпусу, как в бенчмарке. Для быстрой проверки рычага есть --stemmer identity.

Масштаб: BM25 на инвертированном индексе (не брутфорс), пул кандидатов ограничен
(--pool-sample), т.к. прогрев стеммера по всему 825K корпусу — часы.

Защита от false-negatives: из негативов исключаем сам positive_id И пассажи той
же статьи (article_id), берём ранги со --neg-rank-start (не топ-1, чтобы не
подсунуть случайно релевантное).

Вход:  --pairs (query/positive/positive_id), --corpus (KazQAD .jsonl.gz).
Выход: тот же JSONL, но с заполненными "negatives"/"negative_ids".

Пример (быстрый пас без стеммера):
    python scripts/mine_hard_negatives.py \\
        --benchmark-root /content/bench \\
        --pairs data/synthetic_pairs.jsonl \\
        --corpus '/content/KazQAD/data/information-retrieval/corpus/*.jsonl.gz' \\
        --exclude-article-ids data/exclude_article_ids.txt \\
        --stemmer identity --pool-sample 80000 --n-neg 4 \\
        --out data/synthetic_pairs.hn.jsonl

Финальный (твой стеммер):  --stemmer kazakh   (медленно: прогрев кэша через сервис)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import load_corpus, article_id  # noqa: E402


class InvertedBM25:
    """BM25 Okapi на инвертированном индексе — скорит только документы с общими термами."""

    def __init__(self, analyzer, k1: float = 1.5, b: float = 0.75):
        self.analyzer = analyzer
        self.k1, self.b = k1, b
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.freqs: List[Counter] = []
        self.postings: Dict[str, List[int]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0

    def index(self, docs: List[Tuple[str, str]]):
        df: Dict[str, int] = defaultdict(int)
        for did, text in docs:
            toks = self.analyzer(text)
            i = len(self.doc_ids)
            self.doc_ids.append(did)
            fr = Counter(toks)
            self.freqs.append(fr)
            self.doc_len.append(len(toks))
            for t in fr:
                self.postings[t].append(i)
                df[t] += 1
        n = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / n if n else 0.0
        for t, d in df.items():
            self.idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))
        return self

    def search(self, q_terms: List[str], top_k: int) -> List[str]:
        scores: Dict[int, float] = defaultdict(float)
        for t in set(q_terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings.get(t, ()):  # только документы, где терм есть
                tf = self.freqs[i][t]
                dl = self.doc_len[i]
                denom = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + denom)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self.doc_ids[i] for i, _ in ranked]


class KazakhStemmerProd:
    """Production-клиент казахского стеммера: POST /stem/batch, X-API-Key,
    до 1000 слов/запрос, 5 req/сек. План Free: 1000 запросов/мес (=до 1M слов).
    Кэш на диск (стемминг детерминирован → каждое слово запрашивается раз навсегда)."""
    name = "kazakh"
    BASE = "https://kazakh-stemmer-590833642796.europe-west1.run.app"

    def __init__(self, cache_path="data/stem_cache.json", max_per_sec=5, batch=1000,
                 max_requests=0, attempts=5):
        self.key = os.environ.get("KAZAKH_STEMMER_KEY")
        if not self.key:
            raise SystemExit("Нет KAZAKH_STEMMER_KEY в окружении (это X-API-Key стеммера).")
        self.endpoint = self.BASE + "/stem/batch"
        self.cache_path = cache_path
        self.batch = batch
        self.attempts = attempts            # попыток на батч до дробления
        self.max_requests = max_requests    # 0 = без лимита; иначе budget-guard
        self.interval = 1.0 / max_per_sec
        self.cache: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            self.cache = json.load(open(cache_path, encoding="utf-8"))
        self.requests_made = 0
        self.failed_words: List[str] = []

    @staticmethod
    def _trivial(t: str) -> bool:
        return t.isdigit() or len(t) <= 1

    def _post(self, words: List[str]):
        """POST с ретраями и экспоненциальным backoff. 401/403 — фатально;
        429 — дольше ждём; 5xx/сеть/таймаут — повтор. Не глотает ошибку молча:
        если все попытки исчерпаны — поднимает исключение (вызывающий дробит батч)."""
        body = json.dumps({"words": words}).encode("utf-8")
        delay, last = 2.0, None
        for _ in range(self.attempts):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=body, method="POST",
                    headers={"X-API-Key": self.key, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 403):
                    raise SystemExit(f"Стеммер: авторизация отклонена ({e.code}) — "
                                     f"проверь KAZAKH_STEMMER_KEY.")
                if e.code == 429:                       # rate/quota — ждём дольше
                    time.sleep(delay * 2); delay = min(delay * 2, 60); continue
                if 500 <= e.code < 600:                 # серверная — повтор
                    time.sleep(delay); delay = min(delay * 2, 60); continue
                raise                                    # прочие 4xx — не ретраим
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                time.sleep(delay); delay = min(delay * 2, 60); continue
        raise last if last else RuntimeError("стеммер: неизвестный сбой")

    def _save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False)

    def warm(self, tokens):
        uniq = [t for t in set(tokens) if t not in self.cache and not self._trivial(t)]
        for t in set(tokens):
            if self._trivial(t):
                self.cache[t] = t
        n_req = -(-len(uniq) // self.batch)
        print(f"Стеммер (prod): {len(uniq):,} новых уникальных слов → ~{n_req} запросов "
              f"(батч {self.batch}). Кэш: {self.cache_path}")
        if self.max_requests and n_req > self.max_requests:
            raise SystemExit(
                f"Нужно ~{n_req} запросов, а лимит --max-stem-requests={self.max_requests}. "
                f"Уменьши --pool-sample или подними лимит (Free-план стеммера — 1000/мес).")
        # очередь батчей; при сбое батч дробится пополам и возвращается в голову
        queue = [uniq[i:i + self.batch] for i in range(0, len(uniq), self.batch)]
        done = 0
        while queue:
            chunk = queue.pop(0)
            try:
                for it in self._post(chunk):
                    self.cache[it["word"]] = it.get("stem") or it["word"]
                self.requests_made += 1
            except SystemExit:
                raise                                    # фатальная авторизация — наружу
            except Exception as e:
                if len(chunk) > 1:                       # дробим и повторим
                    mid = len(chunk) // 2
                    queue.insert(0, chunk[mid:])
                    queue.insert(0, chunk[:mid])
                    print(f"  [warn] батч упал ({e}) — дроблю {len(chunk)}→2× и повторю")
                    continue
                print(f"  [warn] слово не отстеммилось после ретраев: {chunk[0]!r} — identity")
                self.cache[chunk[0]] = chunk[0]
                self.failed_words.append(chunk[0])
            done += 1
            if done % 20 == 0:
                self._save()
                print(f"  …запросов {self.requests_made}, кэш {len(self.cache):,}")
            if self.max_requests and self.requests_made >= self.max_requests:
                print(f"  [i] достигнут лимит запросов ({self.max_requests}) — "
                      f"стоп. Резюм добьёт остаток позже (кэш сохранён).")
                break
            time.sleep(self.interval)
        self._save()
        msg = f"Стеммер (prod): запросов сделано {self.requests_made}"
        if self.failed_words:
            msg += f"; не отстеммилось слов: {len(self.failed_words)} (identity)"
        print(msg)

    def stem(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        if self._trivial(token):
            return token
        try:                       # не прогрето — одиночный запрос (лучше warm заранее)
            s = (self._post([token])[0].get("stem")) or token
            self.requests_made += 1
        except Exception:
            s = token
        self.cache[token] = s
        return s


def build_analyzer(benchmark_root: str, stemmer_name: str, stem_cache: str,
                   max_stem_requests: int = 0):
    sys.path.insert(0, str(Path(benchmark_root).resolve()))
    from src.preprocess.tokenize import tokenize
    if stemmer_name == "kazakh-prod":
        stemmer = KazakhStemmerProd(cache_path=stem_cache, max_requests=max_stem_requests)

        def analyze(text: str) -> List[str]:
            return [stemmer.stem(t) for t in tokenize(text)]
        return analyze, tokenize, stemmer
    from src.preprocess.stemmer import get_stemmer, stem_tokens
    stemmer = get_stemmer(stemmer_name)

    def analyze(text: str) -> List[str]:
        return stem_tokens(tokenize(text), stemmer)
    return analyze, tokenize, stemmer


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25(+казахский стеммер) hard-negative майнинг")
    ap.add_argument("--benchmark-root", required=True, help="Клон Kaz-RAG-search-benchmark (там твой стеммер).")
    ap.add_argument("--pairs", required=True, help="JSONL с query/positive/positive_id.")
    ap.add_argument("--corpus", required=True, help="Путь/glob к пассажам KazQAD (.jsonl[.gz]).")
    ap.add_argument("--exclude-article-ids", default=None, help="Статьи бенчмарка (антилик пула).")
    ap.add_argument("--stemmer", choices=["identity", "kazakh", "kazakh-prod"],
                    default="identity",
                    help="identity — быстро; kazakh — demo-сервис (30 req/min); "
                         "kazakh-prod — production (/stem/batch, X-API-Key из KAZAKH_STEMMER_KEY).")
    ap.add_argument("--stem-cache", default="data/stem_cache.json",
                    help="Файл кэша стемминга (слово→корень), резюмируемо между прогонами.")
    ap.add_argument("--max-stem-requests", type=int, default=0,
                    help="Budget-guard: макс. запросов к стеммеру (0 — без лимита). "
                         "Free-план стеммера — 1000/мес; при превышении warm остановится, "
                         "резюм добьёт позже.")
    ap.add_argument("--pool-sample", type=int, default=80000,
                    help="Размер пула кандидатов-негативов (позитивы включаются всегда).")
    ap.add_argument("--n-neg", type=int, default=4, help="Сколько негативов на запрос.")
    ap.add_argument("--neg-rank-start", type=int, default=2,
                    help="С какого ранга BM25 брать (0 — топ; ставим ≥2 против false-neg).")
    ap.add_argument("--top-k", type=int, default=50, help="Глубина BM25-выдачи для отбора.")
    ap.add_argument("--out", default="data/pairs.hn.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    analyze, tokenize, stemmer = build_analyzer(
        args.benchmark_root, args.stemmer, args.stem_cache, args.max_stem_requests)

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    # синтетика зовёт поле passage_id; prepare_data — positive_id. Нормализуем.
    for p in pairs:
        if not p.get("positive_id"):
            p["positive_id"] = p.get("passage_id", "")
    print(f"Пар: {len(pairs):,}")

    files = sorted(glob.glob(os.path.expanduser(args.corpus)))
    print("Загрузка корпуса …")
    corpus = load_corpus([Path(f) for f in files])
    exclude = set()
    if args.exclude_article_ids and os.path.exists(args.exclude_article_ids):
        exclude = {l.strip() for l in open(args.exclude_article_ids) if l.strip()}

    # пул: все позитивы пар + случайная выборка корпуса (минус статьи бенчмарка)
    pos_ids = {p["positive_id"] for p in pairs if p.get("positive_id") in corpus}
    sample_src = [d for d in corpus if article_id(d) not in exclude and d not in pos_ids]
    random.shuffle(sample_src)
    pool_ids = list(pos_ids) + sample_src[: max(0, args.pool_sample - len(pos_ids))]
    print(f"Пул кандидатов: {len(pool_ids):,} (позитивов {len(pos_ids):,})")

    # прогрев стеммера (для kazakh — заполняет кэш через сервис, резюмируемо)
    if hasattr(stemmer, "warm"):
        uniq = set()
        for did in pool_ids:
            uniq.update(tokenize(corpus[did]["text"]))
        for p in pairs:
            uniq.update(tokenize(p["query"]))
        print(f"Прогрев стеммера: {len(uniq):,} уникальных токенов (30 req/min — это надолго)…")
        stemmer.warm(uniq)

    print("Индексация BM25 (инвертированный)…")
    bm = InvertedBM25(analyze).index([(did, corpus[did]["text"]) for did in pool_ids])

    added = 0
    n_with = 0
    for p in pairs:
        pid = p.get("positive_id", "")
        pos_art = article_id(pid)
        ranked = bm.search(analyze(p["query"]), args.top_k)
        negs, neg_ids = [], []
        for rank, did in enumerate(ranked):
            if rank < args.neg_rank_start:
                continue
            if did == pid or article_id(did) == pos_art:   # защита от false-neg
                continue
            negs.append(corpus[did]["text"])
            neg_ids.append(did)
            if len(negs) >= args.n_neg:
                break
        p["negatives"], p["negative_ids"] = negs, neg_ids
        added += len(negs)
        n_with += 1 if negs else 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nНегативов добавлено: {added:,} | пар с негативами: {n_with:,}/{len(pairs):,}")
    print(f"Сохранено → {out}")
    print("Дальше: train.py на этом файле (негативы развернутся в триплеты).")


if __name__ == "__main__":
    main()
