"""Classical baselines: TF-IDF, BM25 and personalized PageRank over the candidate set."""
import math
from collections import Counter

import numpy as np

from .pipeline import context_tokens


def _docs(kb, cands):
    return [kb.text_tokens(c) for c in cands]


def tfidf_scores(ins, kb, window=20):
    docs = _docs(kb, ins.candidates)
    ctx, _ = context_tokens(ins.text, window)
    n = len(docs)
    df = Counter(t for d in docs for t in set(d))
    idf = lambda t: math.log((1 + n) / (1 + df[t])) + 1.0
    q = Counter(ctx)
    qv = {t: c * idf(t) for t, c in q.items()}
    out = []
    for d in docs:
        dv = {t: c * idf(t) for t, c in Counter(d).items()}
        num = sum(qv[t] * dv.get(t, 0.0) for t in qv)
        den = math.sqrt(sum(v * v for v in qv.values())) * math.sqrt(sum(v * v for v in dv.values())) or 1.0
        out.append(num / den)
    return np.array(out)


def bm25_scores(ins, kb, window=20, k1=1.5, b=0.75):
    docs = _docs(kb, ins.candidates)
    ctx, _ = context_tokens(ins.text, window)
    n = len(docs)
    avg = (sum(len(d) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf = Counter(d); s = 0.0
        for t in set(ctx):
            if tf[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return np.array(out)


def pagerank_scores(ins, kb, damping=0.85, iters=50):
    """Personalized PageRank on the subgraph of candidates and their neighbors; the restart vector is
    proportional to the token overlap between each concept text and the MeSH terms of the document."""
    nodes = sorted(set(ins.candidates) | {n for c in ins.candidates for n in kb.neighbors(c)})
    idx = {c: i for i, c in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N))
    for c in nodes:
        for n, w in kb.neighbors(c).items():
            if n in idx:
                A[idx[c], idx[n]] += w; A[idx[n], idx[c]] += w
    deg = A.sum(1, keepdims=True); deg[deg == 0] = 1.0
    T = A / deg
    mesh_tokens = set(t for term in ins.mesh_terms for t in term.split())
    r = np.array([len(mesh_tokens & set(kb.text_tokens(c))) for c in nodes], dtype=float) + 1e-6
    r /= r.sum()
    p = np.full(N, 1.0 / N)
    for _ in range(iters):
        p = (1 - damping) * r + damping * (T.T @ p)
    return np.array([p[idx[c]] for c in ins.candidates])
