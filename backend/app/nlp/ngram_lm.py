"""Unit 3: statistical N-gram language model (unigram/bigram/trigram).

* sentence probability with add-k (Laplace when k=1) smoothing
* perplexity as the evaluation measure
* next-word prediction with stupid backoff (trigram -> bigram -> unigram)
"""
import math
from collections import Counter, defaultdict

from app.nlp.preprocessing import sentence_tokenize, word_tokenize

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, n: int = 3, k: float = 0.1):
        self.n, self.k = n, k
        self.counts: list[Counter] = [Counter() for _ in range(n + 1)]   # counts[order][ngram tuple]
        self.next_words: dict[tuple, Counter] = defaultdict(Counter)     # context -> next word counts
        self.vocab: set[str] = set()
        self.total_tokens = 0

    @staticmethod
    def tokenize_sentences(text: str) -> list[list[str]]:
        return [[w.lower() for w in word_tokenize(s) if any(c.isalnum() for c in w)]
                for s in sentence_tokenize(text)]

    def fit(self, text: str) -> "NGramLM":
        for sent in self.tokenize_sentences(text):
            if not sent:
                continue
            toks = [BOS] * (self.n - 1) + sent + [EOS]
            self.vocab.update(sent)
            self.total_tokens += len(sent) + 1
            for order in range(1, self.n + 1):
                for i in range(len(toks) - order + 1):
                    gram = tuple(toks[i:i + order])
                    self.counts[order][gram] += 1
                    if order > 1:
                        self.next_words[gram[:-1]][gram[-1]] += 1
        self.vocab.add(EOS)
        return self

    # ---- probability ----
    def prob(self, word: str, context: tuple) -> float:
        """P(word | context) with add-k smoothing on the highest order."""
        context = context[-(self.n - 1):]
        V = len(self.vocab)
        num = self.counts[len(context) + 1][context + (word,)] + self.k
        den = (self.counts[len(context)][context] if context else self.total_tokens) + self.k * V
        return num / den if den else 1.0 / max(V, 1)

    def sentence_logprob(self, sentence: str) -> tuple[float, int]:
        toks = [w.lower() for w in word_tokenize(sentence) if any(c.isalnum() for c in w)]
        seq = [BOS] * (self.n - 1) + toks + [EOS]
        lp = 0.0
        for i in range(self.n - 1, len(seq)):
            lp += math.log(self.prob(seq[i], tuple(seq[i - self.n + 1:i])))
        return lp, len(toks) + 1

    def sentence_probability(self, sentence: str) -> dict:
        lp, n = self.sentence_logprob(sentence)
        return {"log_prob": round(lp, 3), "probability": math.exp(lp), "perplexity": round(math.exp(-lp / n), 2)}

    def perplexity(self, text: str) -> float:
        lp_total, n_total = 0.0, 0
        for sent in sentence_tokenize(text):
            lp, n = self.sentence_logprob(sent)
            lp_total, n_total = lp_total + lp, n_total + n
        return round(math.exp(-lp_total / n_total), 2) if n_total else float("inf")

    # ---- prediction ----
    def predict_next(self, prefix: str, top: int = 5) -> list[dict]:
        """Stupid backoff: use the longest context that was seen in training."""
        toks = [w.lower() for w in word_tokenize(prefix) if any(c.isalnum() for c in w)]
        ctx = tuple(([BOS] * (self.n - 1) + toks)[-(self.n - 1):])
        weight = 1.0
        while ctx:
            cands = self.next_words.get(ctx)
            if cands:
                total = sum(cands.values())
                return [{"word": w, "prob": round(weight * c / total, 4), "order": len(ctx) + 1}
                        for w, c in cands.most_common(top) if w != EOS][:top]
            ctx, weight = ctx[1:], weight * 0.4
        uni = Counter({g[0]: c for g, c in self.counts[1].items() if g[0] not in (BOS, EOS)})
        return [{"word": w, "prob": round(0.4 * c / self.total_tokens, 4), "order": 1}
                for w, c in uni.most_common(top)]

    def generate(self, prefix: str, length: int = 12) -> str:
        out = prefix.strip()
        for _ in range(length):
            nxt = self.predict_next(out, top=1)
            if not nxt:
                break
            out += " " + nxt[0]["word"]
        return out

    def stats(self) -> dict:
        return {"n": self.n, "k": self.k, "vocab_size": len(self.vocab), "tokens": self.total_tokens,
                "unique_bigrams": len(self.counts[2]), "unique_trigrams": len(self.counts[3]) if self.n >= 3 else 0}
