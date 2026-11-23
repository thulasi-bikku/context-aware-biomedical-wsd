"""Evaluation: accuracy, macro/micro-F1, MRR, recall@K, ECE, bootstrap CI, paired tests."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def ranks_from_scores(scores, gold_idx):
    order = np.argsort(-np.asarray(scores), kind="stable")
    return int(np.where(order == gold_idx)[0][0]) + 1 if gold_idx >= 0 else None


def accuracy(correct):
    return float(np.mean(correct))


def mrr(ranks, absent_as_zero=True):
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks]))


def recall_at_k(ranks, k):
    return float(np.mean([1.0 if (r and r <= k) else 0.0 for r in ranks]))


def macro_f1_by_entity(entities, gold, pred):
    """Average of per-entity macro-F1 (each ambiguous entity weighted equally)."""
    out = []
    for e in sorted(set(entities)):
        idx = [i for i, x in enumerate(entities) if x == e]
        out.append(f1_score([gold[i] for i in idx], [pred[i] for i in idx], average="macro", zero_division=0))
    return float(np.mean(out))


def micro_f1(gold, pred):
    return float(f1_score(gold, pred, average="micro", zero_division=0))


def expected_calibration_error(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def precision_coverage(conf, correct):
    """Precision of the accepted predictions as the confidence threshold is lowered."""
    o = np.argsort(-np.asarray(conf)); c = np.asarray(correct, dtype=float)[o]
    cov = np.arange(1, len(c) + 1) / len(c)
    return cov, np.cumsum(c) / np.arange(1, len(c) + 1)


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); v = np.asarray(values, dtype=float)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test on two aligned score vectors (e.g. per-entity accuracy)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired instance-level correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    return 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
