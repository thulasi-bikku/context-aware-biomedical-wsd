"""Phase 2: recursive probabilistic concept modeling (Eqs. 11-17)."""
import numpy as np
from .text import tokenize


class Vocab:
    def __init__(self, tokens):
        self.itos = sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def distribution(tokens, vocab: Vocab, smoothing: float, weights=None):
    """Additively smoothed distribution over the vocabulary (optionally with token weights)."""
    v = np.full(len(vocab), smoothing, dtype=np.float64)
    for i, t in enumerate(tokens):
        j = vocab.stoi.get(t)
        if j is not None:
            v[j] += 1.0 if weights is None else weights[i]
    return v / v.sum()


def transition(neighbors: dict, kappa: float):
    """A(c'|c) = softmax(s(c,c')/kappa) over the neighbors (Eqs. 11-12). Returns (ids, probs)."""
    if not neighbors:
        return [], np.array([])
    ids = list(neighbors)
    s = np.array([neighbors[c] for c in ids], dtype=np.float64) / kappa
    s -= s.max()
    p = np.exp(s)
    return ids, p / p.sum()


def propagate(cui, kb, vocab: Vocab, l_max=2, kappa=1.0, nu=0.01):
    """Return an array (l_max+1, |V|) holding P_0..P_Lmax for a concept (Eq. 16).

    `kb` must provide `text_tokens(cui)` and `neighbors(cui)` -> {cui: relation score}."""
    base = {}

    def p0(c):
        if c not in base:
            base[c] = distribution(kb.text_tokens(c), vocab, nu)
        return base[c]

    memo = {}

    def pl(c, l):
        if l == 0:
            return p0(c)
        key = (c, l)
        if key not in memo:
            ids, probs = transition(kb.neighbors(c), kappa)
            memo[key] = pl(c, l - 1) if not ids else sum(a * pl(n, l - 1) for n, a in zip(ids, probs))
        return memo[key]

    return np.stack([pl(cui, l) for l in range(l_max + 1)])


def mixture(layers: np.ndarray, lam):
    lam = np.asarray(lam, dtype=np.float64)
    lam = lam / lam.sum()
    return lam @ layers


def s_prob(p_d: np.ndarray, p_ctx: np.ndarray, floor=1e-12):
    """Negative cross-entropy -H(P_d, P_ctx) (Eq. 13)."""
    return float((p_d * np.log(np.maximum(p_ctx, floor))).sum())


def kl(p, q, floor=1e-12):
    return float((p * (np.log(np.maximum(p, floor)) - np.log(np.maximum(q, floor)))).sum())


def default_lambda(l_max, r=0.5):
    w = np.array([(1 - r) * r ** l for l in range(l_max + 1)])
    return w / w.sum()
