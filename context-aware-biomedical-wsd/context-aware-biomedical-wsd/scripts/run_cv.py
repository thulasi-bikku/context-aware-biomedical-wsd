#!/usr/bin/env python
"""Entity-stratified cross-validation on MSH WSD: proposed framework and classical baselines.

Example:
  python scripts/run_cv.py --msh-dir /data/MSHCorpus --mesh-json /data/mesh_by_pmid.json \
      --umls-meta /data/umls/META --folds 10 --out results/
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cawsd import baselines, metrics
from cawsd.data import attach_mesh, entity_stratified_folds, load_msh_wsd
from cawsd.kb import ConceptKB
from cawsd.pipeline import EntityFeaturizer, PipelineConfig


def evaluate(name, rows):
    ent = [r["entity"] for r in rows]; gold = [r["gold"] for r in rows]; pred = [r["pred"] for r in rows]
    ranks = [r["rank"] for r in rows]; correct = [int(r["pred"] == r["gold"]) for r in rows]
    res = {"method": name, "n": len(rows), "accuracy": metrics.accuracy(correct),
           "macro_f1": metrics.macro_f1_by_entity(ent, gold, pred), "micro_f1": metrics.micro_f1(gold, pred),
           "mrr": metrics.mrr(ranks), "recall@1": metrics.recall_at_k(ranks, 1), "recall@5": metrics.recall_at_k(ranks, 5)}
    res["accuracy_ci95"] = metrics.bootstrap_ci(correct)
    if rows and "conf" in rows[0]:
        res["ece"] = metrics.expected_calibration_error([r["conf"] for r in rows], correct)
    return res, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msh-dir", required=True); ap.add_argument("--mesh-json", required=True)
    ap.add_argument("--umls-meta", required=True); ap.add_argument("--candidates")
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--out", default="results")
    ap.add_argument("--no-neural", action="store_true", help="skip PyTorch training (Phases 1-2 scores only)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    data = attach_mesh(load_msh_wsd(a.msh_dir, a.candidates), a.mesh_json)
    seeds = {c for ins in data for c in ins.candidates} | {ins.gold for ins in data}
    kb = ConceptKB.from_rrf(a.umls_meta, seeds)
    folds = entity_stratified_folds(data, a.folds, a.seed)
    cfg = PipelineConfig()
    rows = defaultdict(list)

    for k in range(a.folds):
        test = [i for i, f in enumerate(folds) if f == k]
        dev = [i for i, f in enumerate(folds) if f != k]
        val = set(dev[:: max(1, len(dev) // max(1, len(test)))][: len(test)])   # held-out subset of the development folds
        train = [i for i in dev if i not in val]
        feats = {}
        for ent in sorted({data[i].entity for i in dev}):
            tr = [data[i] for i in train if data[i].entity == ent]
            feats[ent] = EntityFeaturizer(kb, cfg).fit(tr)
        # Phase 2 only (probabilistic score) and baselines
        for i in test:
            ins = data[i]; f = feats.get(ins.entity)
            if f is None:
                continue
            ex = f.featurize(ins)
            from cawsd.graph import mixture, default_lambda, s_prob
            lam = default_lambda(cfg.l_max)
            sc = np.array([s_prob(ex.p_d, mixture(ex.layers[c], lam)) for c in range(len(f.cands))])
            scores = {"phase2_prob": sc,
                      "tfidf": baselines.tfidf_scores(ins, kb), "bm25": baselines.bm25_scores(ins, kb),
                      "pagerank": baselines.pagerank_scores(ins, kb)}
            for name, s in scores.items():
                cands = f.cands if name == "phase2_prob" else ins.candidates
                g = cands.index(ins.gold) if ins.gold in cands else -1
                rows[name].append({"entity": ins.entity, "gold": ins.gold, "pred": cands[int(np.argmax(s))],
                                   "rank": metrics.ranks_from_scores(s, g)})
        if not a.no_neural:
            from cawsd.model import FusionModel, train as fit_model
            tr_ex = [feats[data[i].entity].featurize(data[i]) for i in train if data[i].entity in feats]
            va_ex = [feats[data[i].entity].featurize(data[i]) for i in val if data[i].entity in feats]
            model = fit_model(FusionModel(cfg.n_buckets, 200, cfg.l_max, 0.3), tr_ex, va_ex, epochs=a.epochs, seed=a.seed)
            for i in test:
                ins = data[i]; f = feats.get(ins.entity)
                if f is None:
                    continue
                ex = f.featurize(ins); p = model.predict_proba(ex)
                g = ex.gold
                rows["proposed"].append({"entity": ins.entity, "gold": ins.gold, "pred": f.cands[int(np.argmax(p))],
                                         "rank": metrics.ranks_from_scores(p, g), "conf": float(p.max())})
        print(f"fold {k + 1}/{a.folds} done", flush=True)

    summary, correct = [], {}
    for name, r in rows.items():
        res, c = evaluate(name, r); summary.append(res); correct[name] = c
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if "proposed" in correct:
        for b in ("tfidf", "bm25", "pagerank"):
            if b in correct and len(correct[b]) == len(correct["proposed"]):
                print(f"McNemar proposed vs {b}: p = {metrics.mcnemar_exact(correct['proposed'], correct[b]):.3g}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
