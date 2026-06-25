# 🧬 AI Drug Discovery Platform

> **An AI-powered Virtual Screening Platform for Drug Discovery**
> **B.Tech Biotechnology Final Year Project**

---

## 📖 Overview

The **AI Drug Discovery Platform** is an interactive web application developed using **Python** and **Streamlit** to demonstrate the computational stages of modern drug discovery.

The platform integrates real biological data from public scientific databases with educational virtual screening algorithms, allowing users to explore protein targets, analyze candidate compounds, evaluate physicochemical properties, visualize molecular structures, and generate professional scientific reports.

> **Educational Notice:** This application demonstrates computational workflows for educational and research purposes. Predicted scores are simulated unless generated using validated docking software.

---

# ✨ Features

## 🧬 Protein Analysis

* Search proteins using PDB ID
* Retrieve live protein metadata from the RCSB Protein Data Bank
* Protein name
* Organism
* Experimental method
* Resolution
* Chains
* Co-crystallized ligand
* Active site information
* Protein classification

---

## 💊 Compound Library

Display comprehensive compound information including:

* Compound Name
* Molecular Formula
* Molecular Weight
* Canonical SMILES
* PubChem CID
* DrugBank ID
* 2D Structure
* 3D Structure

---

## 🧪 Virtual Screening

Educational virtual screening pipeline with:

* Compound ranking
* Predicted binding affinity
* Score comparison
* Best candidate identification
* Statistical analysis

---

## 📊 ADMET Analysis

Automatically calculates:

* Molecular Weight
* LogP
* Hydrogen Bond Donors
* Hydrogen Bond Acceptors
* Rotatable Bonds
* Topological Polar Surface Area (TPSA)
* Lipinski Rule of Five

---

## 📈 Scientific Analytics

Interactive visualizations including:

* Score Distribution
* Molecular Weight Distribution
* Lipinski Compliance
* Property Correlation Matrix
* Radar Comparison Charts
* Screening Statistics

---

## 🧬 3D Molecular Visualization

Interactive visualization of:

* Protein Structure
* Ligand Structure
* Protein–Ligand Complex
* Active Site

---

## 📄 Report Generation

Generate downloadable scientific reports containing:

* Protein Information
* Compound Ranking
* ADMET Properties
* Molecular Structures
* Statistical Summary
* Scientific Disclaimer

---

## 🌐 External Database Integration

The application integrates data from:

* RCSB Protein Data Bank
* PubChem
* DrugBank

---

# 🛠 Technology Stack

## Frontend

* Streamlit

## Backend

* Python 3.11

## Scientific Libraries

* Pandas
* NumPy
* RDKit
* Py3Dmol
* Requests
* Plotly
* Matplotlib

## APIs

* RCSB PDB API
* PubChem API
* DrugBank

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 🧬 Supported Protein Targets

| PDB ID | Target                              |
| ------ | ----------------------------------- |
| 1TSR   | Human Thyroid Hormone Receptor Beta |
| 2HU4   | Influenza H5N1 Neuraminidase        |
| 6LU7   | SARS-CoV-2 Main Protease            |
| 1HVR   | HIV-1 Protease                      |
| 3POZ   | PI3K Kinase                         |
| 1ATP   | cAMP-Dependent Protein Kinase       |

---

# 📊 Workflow

1. Enter a Protein Data Bank (PDB) ID
2. Retrieve protein information
3. Load compound library
4. Perform virtual screening
5. Analyze ADMET properties
6. Visualize molecular structures
7. Compare candidate compounds
8. Export scientific report

---

# 📂 Project Structure

```text
AI-Drug-Discovery-Platform/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── packages.txt
├── assets/
├── data/
├── exports/
├── reports/
├── utils/
└── components/
```

---

# ⚠ Scientific Disclaimer

This software is intended exclusively for educational, research, and demonstration purposes.

Protein structures are retrieved from the RCSB Protein Data Bank and compound information is obtained from publicly available scientific databases.

Virtual screening scores presented by this application are computational predictions and must not be interpreted as experimentally validated molecular docking results or evidence of therapeutic efficacy.

Any scientific conclusions should be supported through validated computational methods (e.g., AutoDock Vina) and experimental laboratory studies.

---

# 🔮 Future Enhancements

* AutoDock Vina integration
* Molecular Dynamics Simulation (OpenMM)
* AI-based Lead Prediction
* ChemBERTa Integration
* Graph Neural Networks (GNN)
* ADMETLab Integration
* Protein–Ligand Interaction Maps
* Batch Virtual Screening
* Deep Learning Binding Affinity Prediction
* Cloud Deployment

---

# 👨‍💻 Developer

**Vamsi Krishna Reddy**

B.Tech Biotechnology

AI Drug Discovery Platform

2026

---

## ⭐ If you found this project useful, consider giving it a star.
