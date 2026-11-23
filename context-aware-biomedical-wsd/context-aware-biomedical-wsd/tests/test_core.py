import math
import unittest

import numpy as np

from cawsd import graph, metrics
from cawsd.data import Instance, entity_stratified_folds
from cawsd.kb import ConceptKB
from cawsd.mesh_filter import FilterConfig, fit_filter, jsd_term
from cawsd.pipeline import EntityFeaturizer, PipelineConfig
from cawsd.text import char_ngrams, ngram_ids, tokenize


def toy_kb():
    names = {"C1": ["cold temperature", "low temperature"], "C2": ["common cold", "viral infection of the nose"],
             "C3": ["temperature"], "C4": ["infection", "virus"]}
    rels = {"C1": {"C3": 1.0}, "C2": {"C4": 1.0, "C3": 0.5}, "C3": {"C1": 1.0}, "C4": {"C2": 1.0}}
    return ConceptKB(names, {}, rels)


def toy_data(n=40):
    out = []
    for i in range(n):
        if i % 2:
            out.append(Instance("cold", str(i), "the <e>cold</e> virus caused rhinitis", "cold", "C2", ["C1", "C2"],
                                ["common cold", "virus infection", "rhinitis"]))
        else:
            out.append(Instance("cold", str(i), "exposure to <e>cold</e> lowered the temperature", "cold", "C1", ["C1", "C2"],
                                ["cold temperature", "body temperature"]))
    return out


class TestPhase1(unittest.TestCase):
    def test_term_jsd_nonnegative_and_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b = rng.random(2) * 0.5 + 1e-6
            self.assertGreaterEqual(jsd_term(a, b), -1e-15)
        self.assertAlmostEqual(jsd_term(0.2, 0.2), 0.0)

    def test_weights_normalized(self):
        pos = [["a", "b"], ["a", "c"], ["a"]] * 3
        neg = [["b", "d"], ["d"], ["d", "c"]] * 3
        w = fit_filter(pos, neg, FilterConfig(f_min=1))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertTrue(all(v >= 0 for v in w.values()))
        self.assertIn("a", w); self.assertNotIn("d", w)

    def test_empty_evidence(self):
        self.assertEqual(fit_filter([["a"]], [["a"]], FilterConfig(f_min=1)), {})


class TestPhase2(unittest.TestCase):
    def setUp(self):
        self.kb = toy_kb()
        self.vocab = graph.Vocab([t for c in ("C1", "C2", "C3", "C4") for t in self.kb.text_tokens(c)])

    def test_layers_normalized(self):
        L = graph.propagate("C2", self.kb, self.vocab, l_max=2, kappa=1.0, nu=0.01)
        self.assertEqual(L.shape, (3, len(self.vocab)))
        np.testing.assert_allclose(L.sum(1), 1.0, atol=1e-9)

    def test_mixture_and_kl_equivalence(self):
        lam = graph.default_lambda(2)
        pd = graph.distribution(["virus", "infection"], self.vocab, 0.01)
        sc, kls = [], []
        for c in ("C1", "C2"):
            pc = graph.mixture(graph.propagate(c, self.kb, self.vocab, 2), lam)
            self.assertAlmostEqual(pc.sum(), 1.0)
            sc.append(graph.s_prob(pd, pc)); kls.append(graph.kl(pd, pc))
        self.assertEqual(int(np.argmax(sc)), int(np.argmin(kls)))
        self.assertEqual(int(np.argmax(sc)), 1)      # virus context favours the 'common cold' concept

    def test_transition_softmax(self):
        ids, p = graph.transition({"a": 1.0, "b": 2.0}, 0.5)
        self.assertAlmostEqual(p.sum(), 1.0); self.assertGreater(p[1], p[0])


class TestFeaturesAndMetrics(unittest.TestCase):
    def test_featurizer_shapes(self):
        data = toy_data()
        f = EntityFeaturizer(toy_kb(), PipelineConfig(filter=FilterConfig(f_min=1), n_buckets=5000)).fit(data[:30])
        ex = f.featurize(data[31])
        C = len(f.cands)
        self.assertEqual(ex.layers.shape[0], C); self.assertAlmostEqual(ex.p_d.sum(), 1.0)
        self.assertEqual(len(ex.cov), C); self.assertGreaterEqual(ex.gold, 0)

    def test_ngrams_deterministic(self):
        self.assertEqual(char_ngrams("ab", 3, 3), ["<ab", "ab>"])
        self.assertEqual(ngram_ids("cold", 1000), ngram_ids("cold", 1000))
        self.assertEqual(tokenize("Type-2 diabetes"), ["type-2", "diabetes"])

    def test_folds_cover_all_and_balance(self):
        data = toy_data(40)
        folds = entity_stratified_folds(data, 10)
        self.assertEqual(sorted(set(folds)), list(range(10)))
        self.assertTrue(all(folds.count(k) == 4 for k in range(10)))

    def test_metrics(self):
        self.assertEqual(metrics.ranks_from_scores([0.1, 0.9, 0.5], 2), 2)
        self.assertAlmostEqual(metrics.mrr([1, 2, 4]), (1 + 0.5 + 0.25) / 3)
        self.assertAlmostEqual(metrics.expected_calibration_error([1.0, 1.0], [1, 1]), 0.0)
        lo, hi = metrics.bootstrap_ci([1, 0, 1, 1, 0, 1] * 20); self.assertLess(lo, hi)
        self.assertEqual(metrics.mcnemar_exact([1, 1, 0], [1, 1, 0]), 1.0)
        self.assertAlmostEqual(metrics.macro_f1_by_entity(["a", "a", "b", "b"], [1, 2, 1, 2], [1, 2, 1, 2]), 1.0)


class TestBaselines(unittest.TestCase):
    def test_baselines_rank_virus_context(self):
        from cawsd import baselines
        ins = toy_data()[1]
        kb = toy_kb()
        for fn in (baselines.tfidf_scores, baselines.bm25_scores):
            s = fn(ins, kb); self.assertEqual(ins.candidates[int(np.argmax(s))], "C2")
        self.assertEqual(len(baselines.pagerank_scores(ins, kb)), 2)


if __name__ == "__main__":
    unittest.main()
