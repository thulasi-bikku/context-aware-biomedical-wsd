"""Phase 1: discriminative MeSH-term filtering (Eqs. 3-10)."""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class FilterConfig:
    alpha: float = 1.0       # additive smoothing
    eps: float = 1e-10       # numerical floor in the informativeness ratio
    f_min: int = 2           # minimum frequency
    tau_ir: float = 0.0      # informativeness-ratio threshold
    tau_jsd: float = 0.0     # term-level JSD threshold


def smoothed(counter: Counter, vocab: List[str], alpha: float):
    total = sum(counter[t] for t in vocab) + alpha * len(vocab)
    return {t: (counter[t] + alpha) / total for t in vocab}


def jsd_term(pp: float, pn: float) -> float:
    q = 0.5 * (pp + pn)
    return 0.5 * pp * math.log(pp / q) + 0.5 * pn * math.log(pn / q)


def fit_filter(pos_docs: Iterable[List[str]], neg_docs: Iterable[List[str]], cfg: FilterConfig = FilterConfig()) -> Dict[str, float]:
    """Return {term: weight} for the retained set F, with weights summing to one (empty dict if no mass)."""
    fp, fn = Counter(), Counter()
    for d in pos_docs:
        fp.update(set(d))
    for d in neg_docs:
        fn.update(set(d))
    vocab = sorted(set(fp) | set(fn))
    if not vocab:
        return {}
    pp, pn = smoothed(fp, vocab, cfg.alpha), smoothed(fn, vocab, cfg.alpha)
    raw = {}
    for t in vocab:
        f = fp[t] + fn[t]
        ir = math.log((pp[t] + cfg.eps) / (pn[t] + cfg.eps))
        j = jsd_term(pp[t], pn[t])
        if f >= cfg.f_min and ir >= cfg.tau_ir and j >= cfg.tau_jsd:
            w = max(ir, 0.0) * j
            raw[t] = w
    z = sum(raw.values())
    return {t: w / z for t, w in raw.items() if w > 0} if z > 0 else {}


def coverage(doc_terms: Iterable[str], weights: Dict[str, float]) -> float:
    """Cov(c, F): weighted share of the filtered terms of a concept that occur in the document."""
    s = set(doc_terms)
    return sum(w for t, w in weights.items() if t in s)
