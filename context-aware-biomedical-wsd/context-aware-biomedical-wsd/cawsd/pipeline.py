"""Feature construction for one ambiguous entity: Phase 1 filters, Phase 2 distributions, n-gram inputs."""
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .graph import Vocab, distribution, propagate
from .mesh_filter import FilterConfig, coverage, fit_filter
from .text import ngram_ids, tokenize


@dataclass
class PipelineConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    l_max: int = 2
    kappa: float = 1.0
    mu: float = 0.01         # document-distribution smoothing
    nu: float = 0.01         # concept-distribution smoothing
    window: int = 20         # context tokens on each side of the mention
    n_min: int = 3
    n_max: int = 5
    n_buckets: int = 100_000
    max_terms: int = 200     # cap on |T(c)|


@dataclass
class Example:
    entity: str
    cands: List[str]
    gold: int                      # index of the gold concept in cands (-1 if unknown / absent)
    p_d: np.ndarray                # (V,)
    layers: np.ndarray             # (C, L+1, V)
    cov: np.ndarray                # (C,)
    logprior: np.ndarray           # (C,)
    ctx_ids: List[List[int]]       # n-gram ids per context token
    mention_ids: List[int]
    cand_term_ids: List[List[List[int]]]   # per candidate: n-gram ids per term


def context_tokens(text, window):
    left, _, rest = text.partition("<e>")
    mention, _, right = rest.partition("</e>")
    lt, rt = tokenize(left), tokenize(right)
    return lt[-window:] + rt[:window], tokenize(mention)


class EntityFeaturizer:
    """Fit on the training instances of one entity, then featurize any instance of that entity."""

    def __init__(self, kb, cfg: PipelineConfig = PipelineConfig()):
        self.kb, self.cfg = kb, cfg

    def fit(self, train):
        cands = sorted({c for ins in train for c in ins.candidates} | {ins.gold for ins in train})
        self.cands = cands
        by_label = defaultdict(list)
        for ins in train:
            by_label[ins.gold].append(ins.mesh_terms)
        self.filters = {}
        for c in cands:
            pos = by_label.get(c, [])
            neg = [d for k, v in by_label.items() if k != c for d in v]
            self.filters[c] = fit_filter(pos, neg, self.cfg.filter)
        union = set().union(*[set(w) for w in self.filters.values()]) if self.filters else set()
        toks = [t for term in union for t in tokenize(term)]
        for c in cands:
            toks += self.kb.text_tokens(c)
        self.vocab = Vocab(toks or ["<unk>"])
        self.union_terms = union
        self.layers = np.stack([propagate(c, self.kb, self.vocab, self.cfg.l_max, self.cfg.kappa, self.cfg.nu) for c in cands])
        counts = Counter(ins.gold for ins in train)
        n = sum(counts.values())
        self.logprior = np.array([math.log((counts[c] + 1) / (n + len(cands))) for c in cands])
        self.cand_term_ids = [[ngram_ids(t, self.cfg.n_buckets, self.cfg.n_min, self.cfg.n_max)
                               for t in self.kb.term_list(c)[: self.cfg.max_terms]] for c in cands]
        return self

    def featurize(self, ins) -> Example:
        sel = [t for t in ins.mesh_terms if t in self.union_terms]
        dtoks = [x for t in sel for x in tokenize(t)]
        p_d = distribution(dtoks, self.vocab, self.cfg.mu)
        cov = np.array([coverage(ins.mesh_terms, self.filters[c]) for c in self.cands])
        ctx, ment = context_tokens(ins.text, self.cfg.window)
        ctx_ids = [ngram_ids(t, self.cfg.n_buckets, self.cfg.n_min, self.cfg.n_max) for t in ctx] or \
                  [ngram_ids("<unk>", self.cfg.n_buckets)]
        ment_ids = [i for t in ment for i in ngram_ids(t, self.cfg.n_buckets, self.cfg.n_min, self.cfg.n_max)] or \
                   ngram_ids("<unk>", self.cfg.n_buckets)
        gold = self.cands.index(ins.gold) if ins.gold in self.cands else -1
        return Example(ins.entity, self.cands, gold, p_d, self.layers, cov, self.logprior, ctx_ids, ment_ids, self.cand_term_ids)
