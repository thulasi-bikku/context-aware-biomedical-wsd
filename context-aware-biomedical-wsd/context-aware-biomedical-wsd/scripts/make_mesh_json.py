#!/usr/bin/env python
"""Build the PMID -> MeSH descriptor table needed by Phase 1 from a PubMed/MEDLINE XML baseline or
an E-utilities export.  Usage: python scripts/make_mesh_json.py --pmids pmids.txt --xml pubmed_*.xml.gz --out mesh_by_pmid.json"""
import argparse, gzip, json
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", required=True, help="text file with one PMID per line (from term_pmid_cui)")
    ap.add_argument("--xml", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = {l.strip() for l in open(a.pmids) if l.strip()}
    table = {}
    for path in a.xml:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rb") as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag == "MedlineCitation":
                    pmid = el.findtext("PMID")
                    if pmid in want:
                        table[pmid] = [d.text for d in el.findall(".//MeshHeading/DescriptorName") if d.text]
                    el.clear()
    json.dump(table, open(a.out, "w"))
    print(f"{len(table)} of {len(want)} PMIDs found")


if __name__ == "__main__":
    main()
