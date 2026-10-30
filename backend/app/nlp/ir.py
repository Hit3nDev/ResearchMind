"""Unit 5: Information Retrieval from scratch.

* Inverted index construction (term -> postings {doc_id: term frequency})
* Boolean model (AND / OR / NOT, plus coordination-level matching)
* Vector Space Model with TF-IDF weighting and cosine similarity
* Probabilistic model: Okapi BM25
* Evaluation: Precision, Recall, F-measure, Average Precision, MAP
* Reciprocal Rank Fusion to combine the models into one hybrid ranking
"""
import math
import re
from collections import Counter, defaultdict

from app.nlp.preprocessing import preprocess


class InvertedIndex:
    def __init__(self, docs: dict[str, str]):
        self.doc_ids = list(docs)
        self.N = len(docs)
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_len: dict[str, int] = {}
        for did, text in docs.items():
            terms = preprocess(text)
            self.doc_len[did] = len(terms)
            for term, tf in Counter(terms).items():
                self.postings[term][did] = tf
        self.avgdl = (sum(self.doc_len.values()) / self.N) if self.N else 0
        self._doc_norm = {did: math.sqrt(sum(w * w for w in self._doc_vector(did).values())) or 1.0
                          for did in self.doc_ids}

    # ---------- weighting ----------
    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log((self.N + 1) / (df + 1)) + 1          # smoothed idf

    @staticmethod
    def tf_weight(tf: int) -> float:
        return 1 + math.log(tf) if tf > 0 else 0.0           # sub-linear tf scaling

    def _doc_vector(self, did: str) -> dict[str, float]:
        return {t: self.tf_weight(p[did]) * self.idf(t) for t, p in self.postings.items() if did in p}

    # ---------- Boolean model ----------
    def boolean(self, expr: str) -> set[str]:
        """Evaluates e.g. 'crop AND (disease OR pest) AND NOT survey' left-to-right with parentheses."""
        tokens = re.findall(r"\(|\)|AND|OR|NOT|[^\s()]+", expr)
        pos = 0
        universe = set(self.doc_ids)

        def term_docs(word: str) -> set[str]:
            stems = preprocess(word)
            return set(self.postings.get(stems[0], {})) if stems else set()

        def parse_or():
            nonlocal pos
            left = parse_and()
            while pos < len(tokens) and tokens[pos] == "OR":
                pos += 1
                left = left | parse_and()
            return left

        def parse_and():
            nonlocal pos
            left = parse_not()
            while pos < len(tokens) and tokens[pos] not in ("OR", ")"):
                if tokens[pos] == "AND":
                    pos += 1
                left = left & parse_not()       # implicit AND between adjacent terms
            return left

        def parse_not():
            nonlocal pos
            if pos < len(tokens) and tokens[pos] == "NOT":
                pos += 1
                return universe - parse_not()
            if pos < len(tokens) and tokens[pos] == "(":
                pos += 1
                res = parse_or()
                pos += 1  # skip ')'
                return res
            word = tokens[pos] if pos < len(tokens) else ""
            pos += 1
            return term_docs(word)

        return parse_or() if tokens else set()

    def coordination_scores(self, query: str) -> dict[str, float]:
        """Ranked Boolean: fraction of query terms present in the document (OR semantics)."""
        q = set(preprocess(query))
        if not q:
            return {d: 0.0 for d in self.doc_ids}
        return {d: sum(1 for t in q if d in self.postings.get(t, {})) / len(q) for d in self.doc_ids}

    # ---------- Vector Space Model ----------
    def tfidf_scores(self, query: str) -> dict[str, float]:
        qtf = Counter(preprocess(query))
        qvec = {t: self.tf_weight(c) * self.idf(t) for t, c in qtf.items()}
        qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
        scores = dict.fromkeys(self.doc_ids, 0.0)
        for t, qw in qvec.items():
            for did, tf in self.postings.get(t, {}).items():
                scores[did] += qw * self.tf_weight(tf) * self.idf(t)
        return {d: s / (qnorm * self._doc_norm[d]) for d, s in scores.items()}

    # ---------- Probabilistic model ----------
    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        scores = dict.fromkeys(self.doc_ids, 0.0)
        for t in set(preprocess(query)):
            df = self.df(t)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))   # Robertson-Sparck Jones weight
            for did, tf in self.postings[t].items():
                norm = tf + k1 * (1 - b + b * self.doc_len[did] / (self.avgdl or 1))
                scores[did] += idf * tf * (k1 + 1) / norm
        return scores

    def stats(self) -> dict:
        top_terms = sorted(self.postings, key=self.df, reverse=True)[:15]
        return {"documents": self.N, "vocabulary": len(self.postings), "avg_doc_len": round(self.avgdl, 1),
                "sample_postings": {t: self.postings[t] for t in top_terms}}


def rank(scores: dict[str, float]) -> list[str]:
    return [d for d, s in sorted(scores.items(), key=lambda kv: -kv[1])]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, d in enumerate(r):
            fused[d] += 1.0 / (k + i + 1)
    return dict(fused)


# ---------------- Evaluation metrics ----------------
def precision_recall_f(retrieved: list[str], relevant: set[str], beta: float = 1.0) -> dict:
    hits = sum(1 for d in retrieved if d in relevant)
    p = hits / len(retrieved) if retrieved else 0.0
    r = hits / len(relevant) if relevant else 0.0
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f_measure": round(f, 3)}


def average_precision(ranking: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return round(total / len(relevant), 3) if relevant else 0.0


def mean_average_precision(runs: list[tuple[list[str], set[str]]]) -> float:
    return round(sum(average_precision(r, rel) for r, rel in runs) / len(runs), 3) if runs else 0.0
