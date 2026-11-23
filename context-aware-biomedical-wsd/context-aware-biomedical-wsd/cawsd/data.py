"""MSH WSD loading, MeSH attachment and entity-stratified folds."""
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Instance:
    entity: str
    pmid: str
    text: str                 # title + abstract with the target wrapped in <e>...</e>
    mention: str
    gold: str                 # gold CUI
    candidates: List[str]     # candidate CUIs
    mesh_terms: List[str] = field(default_factory=list)


def load_msh_wsd(root, candidates_file=None):
    """Read the official MSHCorpus directory (benchmark_mesh.txt and the *_pmids_tagged.arff files).

    Candidate sets default to the sense inventory of the benchmark. A JSON file mapping
    '<entity>' -> list of CUIs (or '<entity>|<pmid>' -> list of CUIs) can override them, for example
    with MetaMap-generated candidates.
    """
    root = Path(root)
    if (root / "MSHCorpus").is_dir():
        root = root / "MSHCorpus"
    senses = {}
    for line in (root / "benchmark_mesh.txt").read_text(encoding="utf8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) > 1:
            senses[parts[0]] = {f"M{i}": c for i, c in enumerate(parts[1:], 1)}
    override = json.loads(Path(candidates_file).read_text()) if candidates_file else {}
    out = []
    for f in sorted(root.glob("*arff")):
        entity = f.name[: -len("_pmids_tagged.arff")]
        lines = [x.strip() for x in f.read_text(encoding="iso-8859-1").splitlines()]
        start = next(i for i, x in enumerate(lines) if x.startswith("@DATA")) + 1
        for line in lines[start:]:
            m_pmid = re.search(r"[0-9]+(?=,)", line)
            m_label = re.search(r"(?<=,)M[0-9]+", line)
            if not (m_pmid and m_label):
                continue
            text = line[m_pmid.end() + 1: m_label.start() - 1].strip('"')
            mm = re.search(r"<e>(.+?)</e>", text)
            cands = override.get(f"{entity}|{m_pmid.group()}", override.get(entity, list(senses[entity].values())))
            out.append(Instance(entity, m_pmid.group(), text, mm.group(1) if mm else entity,
                                senses[entity][m_label.group()], list(cands)))
    return out


def attach_mesh(instances, mesh_json):
    """Attach MeSH descriptor names per PMID. `mesh_json` maps PMID -> list of descriptor names,
    extracted from MEDLINE/PubMed records (the MSH WSD download contains titles and abstracts only)."""
    table = json.loads(Path(mesh_json).read_text())
    for ins in instances:
        ins.mesh_terms = [t.lower() for t in table.get(ins.pmid, [])]
    return instances


def entity_stratified_folds(instances, n_folds=10, seed=0):
    """Fold index per instance; instances of every entity are spread evenly over the folds."""
    rng = random.Random(seed)
    by_entity: Dict[str, List[int]] = {}
    for i, ins in enumerate(instances):
        by_entity.setdefault(ins.entity, []).append(i)
    fold = [0] * len(instances)
    offset = 0
    for ent in sorted(by_entity):
        idx = by_entity[ent]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold[i] = (k + offset) % n_folds
        offset += len(idx)
    return fold
