# 🧬 Drug Discovery Pipeline
**Virtual Screening Simulation — B.Tech Biotechnology Final Year Project**

---

## What This App Does

An educational Streamlit web application that simulates the virtual screening stage of drug discovery.
It fetches **real protein data** from the RCSB Protein Data Bank and computes **simulated screening scores**
based on compound physicochemical properties.

> ⚠️ **Scores shown are educational simulations and not actual docking results.**

---

## Features

| Feature | Details |
|---|---|
| Live Protein Data | Fetched from RCSB PDB API (name, organism, method, resolution, chains, ligands) |
| Active Site Residues | Curated annotations for key drug targets |
| Reference Ligand | Co-crystallized ligand info (e.g. Oseltamivir for 2HU4) |
| ADMET Panel | MW, LogP, H-Donors, H-Acceptors, Rotatable Bonds, Lipinski Ro5 |
| Compound Details | Name, formula, 2D/3D structure (PubChem), PubChem CID, DrugBank ID, SMILES |
| 3D Protein Viewer | Embedded RCSB / NGL Viewer with ligand & active site highlighting |
| Scientific Metrics | Compound count, avg MW, best/worst/avg scores, distribution chart |
| PDF Report | Protein info · Reference ligand · ADMET · Ranking · Statistics · Disclaimer |
| Future Upgrades | Roadmap to AutoDock Vina, DrugBank API, AI-based prediction, etc. |

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Validated PDB IDs

| PDB ID | Target | Reference Ligand |
|---|---|---|
| 2HU4 | Influenza H5N1 Neuraminidase | Oseltamivir (Tamiflu) |
| 6LU7 | SARS-CoV-2 Main Protease | N3 Inhibitor |
| 1HVR | HIV-1 Protease | XK263 Cyclic Urea |
| 3POZ | PI3K Kinase | — |
| 1ATP | cAMP-Dependent Protein Kinase | — |

---

## Terminology

| Old term | New (accurate) term |
|---|---|
| Docking Simulation | Virtual Screening Simulation |
| Best Candidate | Highest Simulated Score |

---

## Disclaimer

This project is for **educational purposes only**. Protein 3D structures are real data from the
[RCSB Protein Data Bank](https://www.rcsb.org). Simulated scores are calculated from a formula
using compound properties and protein metadata — they **do not** represent results from validated
docking software (e.g. AutoDock Vina) and must not be used for actual drug research or clinical
decisions.

---

## Future Upgrades (Roadmap)

- AutoDock Vina integration for real molecular docking
- DrugBank API for drug interaction data
- PubChem API for live compound property fetching
- AI-based lead prediction (GNN / ChemBERTa)
- Molecular dynamics (MD) simulation via OpenMM
- ADMET prediction via ADMETlab 2.0 API
