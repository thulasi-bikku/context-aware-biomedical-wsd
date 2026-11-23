# Data

The repository does not contain data. MSH WSD and UMLS are distributed under the UMLS license.

1. MSH WSD: download `MSHCorpus.zip` from https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html (UMLS Terminology Services account required) and unzip it. The directory contains `benchmark_mesh.txt` and one `*_pmids_tagged.arff` file per ambiguous word.
2. MeSH headings: the download holds titles and abstracts only. Extract the MeSH descriptor names of every PMID from a PubMed/MEDLINE baseline with `scripts/make_mesh_json.py`.
3. UMLS: download a UMLS release and point `--umls-meta` to its `META` directory (`MRCONSO.RRF`, `MRDEF.RRF`, `MRREL.RRF`).
