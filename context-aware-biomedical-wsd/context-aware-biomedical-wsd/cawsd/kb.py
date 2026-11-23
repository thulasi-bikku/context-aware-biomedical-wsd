"""Concept knowledge base: UMLS RRF loader and an in-memory variant used by tests."""
import csv
from collections import defaultdict
from pathlib import Path
from .text import tokenize

DEFAULT_REL_WEIGHTS = {"PAR": 1.0, "CHD": 1.0, "RB": 1.0, "RN": 1.0, "SY": 1.0, "SIB": 1.0, "RO": 1.0, "RQ": 1.0}


class ConceptKB:
    def __init__(self, names, defs=None, rels=None, mesh_descriptors=None):
        self._names = names            # cui -> list[str]
        self._defs = defs or {}        # cui -> list[str]
        self._rels = rels or {}        # cui -> {cui2: score}
        self._mesh = mesh_descriptors or {}

    def text_tokens(self, cui):
        parts = self._names.get(cui, []) + self._defs.get(cui, []) + self._mesh.get(cui, [])
        return [t for p in parts for t in tokenize(p)]

    def term_list(self, cui):
        """T(c): preferred name, synonyms, definition and MeSH descriptor terms (used by Phase 3)."""
        return sorted(set(self.text_tokens(cui)))

    def neighbors(self, cui):
        return self._rels.get(cui, {})

    @classmethod
    def from_rrf(cls, meta_dir, seed_cuis, rel_weights=None, max_neighbors=50, lang="ENG"):
        """Read MRCONSO/MRDEF/MRREL from a UMLS META directory for the seed concepts and their neighbors."""
        meta = Path(meta_dir)
        w = rel_weights or DEFAULT_REL_WEIGHTS
        seeds = set(seed_cuis)
        rels = defaultdict(lambda: defaultdict(float))
        with open(meta / "MRREL.RRF", encoding="utf8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                c1, rel, c2 = row[0], row[3], row[4]
                if c1 in seeds and c1 != c2 and rel in w:
                    rels[c1][c2] += w[rel]
        keep = {}
        for c, nb in rels.items():
            top = sorted(nb.items(), key=lambda kv: -kv[1])[:max_neighbors]
            keep[c] = dict(top)
        universe = set(seeds)
        for nb in keep.values():
            universe.update(nb)
        names = defaultdict(list); defs = defaultdict(list)
        with open(meta / "MRCONSO.RRF", encoding="utf8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                if row[1] == lang and row[0] in universe and row[16] == "N":
                    names[row[0]].append(row[14])
        with open(meta / "MRDEF.RRF", encoding="utf8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                if row[0] in universe:
                    defs[row[0]].append(row[5])
        return cls({k: sorted(set(v)) for k, v in names.items()}, dict(defs), keep)
