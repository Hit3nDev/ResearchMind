"""Unit 5: word representations.

* One-hot encoding and Bag-of-Words (for comparison)
* Word2Vec trained from scratch in NumPy - Skip-gram or CBOW with negative sampling.
  Training is vectorised over mini-batches of (target, context) pairs so it runs
  in a few seconds on the corpus of retrieved papers.
"""
import numpy as np

from app.nlp.preprocessing import sentence_tokenize, stop_words, word_tokenize


def one_hot(vocab: list[str], word: str) -> list[int]:
    return [1 if v == word else 0 for v in vocab]


def bag_of_words(docs: list[str]) -> dict:
    tokenized = [[w.lower() for w in word_tokenize(d) if w.isalpha()] for d in docs]
    vocab = sorted({w for t in tokenized for w in t})
    return {"vocab": vocab, "vectors": [[t.count(v) for v in vocab] for t in tokenized]}


class Word2Vec:
    def __init__(self, dim: int = 50, window: int = 3, negative: int = 5, min_count: int = 2,
                 epochs: int = 12, lr: float | None = None, mode: str = "skipgram", seed: int = 7):
        self.dim, self.window, self.negative, self.min_count = dim, window, negative, min_count
        # CBOW sees one example per position (vs 2*window for skip-gram), so it needs a larger step
        self.epochs, self.mode = epochs, mode
        self.lr = lr if lr is not None else (0.05 if mode == "skipgram" else 0.3)
        self.rng = np.random.default_rng(seed)
        self.w2i: dict[str, int] = {}
        self.i2w: list[str] = []
        self.losses: list[float] = []

    @staticmethod
    def sentences(text: str) -> list[list[str]]:
        sw = stop_words()
        return [[w.lower() for w in word_tokenize(s) if w.isalpha() and len(w) > 1 and w.lower() not in sw]
                for s in sentence_tokenize(text)]

    def _build_vocab(self, sents):
        counts: dict[str, int] = {}
        for s in sents:
            for w in s:
                counts[w] = counts.get(w, 0) + 1
        self.i2w = sorted((w for w, c in counts.items() if c >= self.min_count), key=lambda w: -counts[w])
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        freq = np.array([counts[w] for w in self.i2w], dtype=float) ** 0.75   # unigram^(3/4) noise distribution
        self.noise = freq / freq.sum()

    def _examples(self, sents) -> tuple[np.ndarray, np.ndarray]:
        """Returns (inputs, outputs). Each input row holds 2*window word ids padded with -1.
        Skip-gram: input = [center], output = one context word (one row per pair).
        CBOW:      input = all context words of a position, output = the center word."""
        W2 = 2 * self.window
        inp, out = [], []
        for s in sents:
            ids = [self.w2i[w] for w in s if w in self.w2i]
            for i, center in enumerate(ids):
                ctx = [ids[j] for j in range(max(0, i - self.window), min(len(ids), i + self.window + 1)) if j != i]
                if not ctx:
                    continue
                if self.mode == "skipgram":
                    for c in ctx:
                        inp.append([center] + [-1] * (W2 - 1))
                        out.append(c)
                else:
                    inp.append(ctx + [-1] * (W2 - len(ctx)))
                    out.append(center)
        return np.array(inp), np.array(out)

    @staticmethod
    def _apply(W: np.ndarray, rows: np.ndarray, grads: np.ndarray) -> None:
        uniq, inv = np.unique(rows, return_inverse=True)
        acc = np.zeros((len(uniq), W.shape[1]))
        np.add.at(acc, inv, grads)
        W[uniq] -= acc / np.bincount(inv)[:, None]

    def fit(self, text: str) -> "Word2Vec":
        sents = self.sentences(text)
        self._build_vocab(sents)
        V = len(self.i2w)
        if V < 5:
            raise ValueError("Corpus too small to train Word2Vec")
        self.W_in = (self.rng.random((V, self.dim)) - 0.5) / self.dim
        self.W_out = np.zeros((V, self.dim))
        inputs, outputs = self._examples(sents)
        batch = 512
        for epoch in range(self.epochs):
            order = self.rng.permutation(len(inputs))
            lr = self.lr * (1 - epoch / self.epochs) + 1e-4
            epoch_loss = 0.0
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                x, y = inputs[idx], outputs[idx]
                neg = self.rng.choice(V, size=(len(idx), self.negative), p=self.noise)
                mask = x >= 0                                      # (B, 2w) real context slots
                cnt = mask.sum(axis=1, keepdims=True)
                h = (self.W_in[np.where(mask, x, 0)] * mask[:, :, None]).sum(axis=1) / cnt   # mean input vector (B, d)
                targets = np.concatenate([y[:, None], neg], axis=1)  # (B, 1+k)
                u = self.W_out[targets]                            # (B, 1+k, d)
                score = np.einsum("bkd,bd->bk", u, h)
                label = np.zeros_like(score)
                label[:, 0] = 1
                sig = 1 / (1 + np.exp(-np.clip(score, -10, 10)))
                g = (sig - label) * lr                             # gradient of logistic loss
                epoch_loss += -np.log(sig[:, 0] + 1e-9).sum() - np.log(1 - sig[:, 1:] + 1e-9).sum()
                # Accumulate gradients, then average per word: frequent words appear many times
                # in one mini-batch and summing their updates would make training diverge.
                grad_h = np.einsum("bk,bkd->bd", g, u) / cnt       # spread equally over the averaged inputs
                rows = np.broadcast_to(np.arange(len(idx))[:, None], x.shape)[mask]
                self._apply(self.W_in, x[mask], grad_h[rows])
                self._apply(self.W_out, targets.ravel(), (g[:, :, None] * h[:, None, :]).reshape(-1, self.dim))
            self.losses.append(round(float(epoch_loss / max(len(order), 1)), 4))
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        self.unit = self.W_in / np.where(norms == 0, 1, norms)
        return self

    def vector(self, word: str) -> list[float] | None:
        i = self.w2i.get(word.lower())
        return None if i is None else [round(float(v), 4) for v in self.W_in[i]]

    def most_similar(self, word: str, top: int = 8) -> list[dict]:
        i = self.w2i.get(word.lower())
        if i is None:
            return []
        sims = self.unit @ self.unit[i]
        best = np.argsort(-sims)[1:top + 1]
        return [{"word": self.i2w[j], "similarity": round(float(sims[j]), 3)} for j in best]

    def projection_2d(self, top: int = 60) -> list[dict]:
        """PCA of the most frequent words' vectors for a scatter plot."""
        n = min(top, len(self.i2w))
        X = self.W_in[:n] - self.W_in[:n].mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        pts = X @ vt[:2].T
        return [{"word": self.i2w[i], "x": round(float(pts[i, 0]), 4), "y": round(float(pts[i, 1]), 4)}
                for i in range(n)]
