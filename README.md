# Context-aware biomedical entity disambiguation

Reference implementation of a three-phase, candidate-restricted framework for biomedical entity disambiguation:

1. **Phase 1, MeSH filtering** (`cawsd/mesh_filter.py`): frequency, informativeness-ratio and term-level Jensen–Shannon criteria with normalized non-negative weights.
2. **Phase 2, probabilistic concept modeling** (`cawsd/graph.py`, `cawsd/kb.py`): temperature-scaled propagation of concept language models over UMLS relations and negative cross-entropy scoring.
3. **Phase 3, character n-gram fusion** (`cawsd/model.py`): hashed character n-gram embeddings, attention over the context, fused candidate logits, candidate-restricted softmax and the joint objective trained with Adam.

Classical baselines (TF-IDF, BM25, personalized PageRank) are in `cawsd/baselines.py`; metrics, bootstrap confidence intervals, paired tests and calibration utilities are in `cawsd/metrics.py`.

## Installation

```
pip install -r requirements.txt
```

## Data

See `data/README.md`. Data cannot be redistributed.

## Running the cross-validation

```
python scripts/run_cv.py --msh-dir /path/MSHCorpus --mesh-json mesh_by_pmid.json \
    --umls-meta /path/UMLS/META --folds 10 --out results/
```

`--no-neural` evaluates Phases 1–2 and the baselines without PyTorch. By default the candidate sets are the sense inventories of `benchmark_mesh.txt`. To use MetaMap-generated candidate sets, as in the manuscript, pass `--candidates` with a JSON file mapping `entity` (or `entity|pmid`) to a list of CUIs; accuracy values then depend on the candidate generator and are not comparable across the two settings.

## Tests

```
python -m unittest discover -s tests -t .
```

The tests cover the properties stated in the manuscript (normalization of the term weights, non-negativity of the term-level divergence, normalization of the propagated distributions, equivalence of cross-entropy and KL ranking) and the evaluation utilities. The PyTorch test is skipped when PyTorch is not installed.

## Notes on the implementation

* The mixture weights over propagation depths are learned through a softmax; the fusion coefficients are learned parameters; the attention query and n-gram embeddings are trained jointly with the supervised and contextual losses.
* The mention vector of Eq. 18 is computed but, as in the manuscript, the embedding score uses the context vector and the candidate vector; `FusionModel(use_mention=True)` adds the mention vector to the context vector.
* Smoothing constants, thresholds and the loss weight are hyperparameters and have to be tuned on the validation subset; the defaults in the code are starting values.

## Citation

Please cite the accompanying manuscript when using this code.

## License

MIT
