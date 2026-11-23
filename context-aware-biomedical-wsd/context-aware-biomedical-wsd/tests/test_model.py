import unittest

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestModel(unittest.TestCase):
    def test_train_and_predict_on_toy_data(self):
        from cawsd.mesh_filter import FilterConfig
        from cawsd.model import FusionModel, train
        from cawsd.pipeline import EntityFeaturizer, PipelineConfig
        from tests.test_core import toy_data, toy_kb
        data = toy_data(60)
        cfg = PipelineConfig(filter=FilterConfig(f_min=1), n_buckets=5000)
        f = EntityFeaturizer(toy_kb(), cfg).fit(data[:40])
        tr = [f.featurize(x) for x in data[:40]]; te = [f.featurize(x) for x in data[40:]]
        m = FusionModel(5000, 16, cfg.l_max, 0.0)
        train(m, tr, te, epochs=5, lr=1e-2, batch_size=8)
        p = m.predict_proba(te[0])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        acc = sum(int(p_.argmax() == e.gold) for e, p_ in ((e, m.predict_proba(e)) for e in te)) / len(te)
        self.assertGreater(acc, 0.8)


if __name__ == "__main__":
    unittest.main()
