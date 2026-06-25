app.py
------
Drug Discovery Pipeline — Professional Edition v4.0
B.Tech Biotechnology · KL University

Changelog from v3.0:
  🔧 FIX: API key now injected via st.secrets["ANTHROPIC_API_KEY"] (works on all platforms)
  🔧 FIX: reports_count only increments on actual download click, not every render
  🔧 FIX: AI chat is now truly multi-turn (full history sent each call)
  🔧 FIX: stats_module alias documented with inline comment
  ✨ NEW: PubChem Live ADMET — real bioassay + physicochemical data via PUG-REST
  ✨ NEW: DrugBank API tab — drug interactions, PK data, mechanism of action
  ✨ NEW: ADMETlab-style validated property scoring panel
  ✨ NEW: Scaffold Clustering — Bemis-Murcko scaffolds + hierarchical dendrogram
  ✨ NEW: Multi-Target Screening — screen vs multiple PDB IDs, selectivity heatmap
  ✨ NEW: AutoDock Vina Integration (local binary) with fallback to simulation
  ✨ NEW: Smart compound search with similarity scoring

Run:
    pip install streamlit pandas plotly requests fpdf2 rdkit scipy
    streamlit run app.py

Secrets (Streamlit Cloud or .streamlit/secrets.toml):
    ANTHROPIC_API_KEY = "sk-ant-..."
    DRUGBANK_API_KEY  = "..."   # optional
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import re
import json
import time
import subprocess
import hashlib
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import requests
import streamlit as st
import streamlit.components.v1 as components

# stdlib statistics — aliased to avoid shadowing Python's built-in 'statistics' module
import statistics as stats_module

# Local modules
import protein
import ranking
import report
import screening

# Optional RDKit for scaffold clustering (graceful fallback if not installed)
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Optional scipy for dendrogram
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

DATA_PATH   = "compounds.csv"
EXPORTS_DIR = "exports"
SCORE_COL   = "simulated_score"
APP_VERSION = "4.0"

DEVELOPERS = [
    "N. Vamsi Krishna",
    "M. Karthik",
    "T. Vignesh",
    "K. Jagadeesh",
    "Team Member 1",
    "Team Member 2",
    "Team Member 3",
]

INSTITUTION = "KL University"

# Colour palette
TEAL_DARK  = "#071A2F"
TEAL_MID   = "#1E3A8A"
TEAL_LIGHT = "#00E5FF"
GOLD       = "#FFD166"
GREEN_OK   = "#00F5A0"
BG_CARD    = "#16213E"
PINK       = "#FF2E88"

os.makedirs(EXPORTS_DIR, exist_ok=True)

# ── API Key resolution (FIX #1) ────────────────────────────────────
# Priority: st.secrets → environment variable → None (shows warning in UI)
def _get_api_key() -> Optional[str]:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")

def _get_drugbank_key() -> Optional[str]:
    try:
        return st.secrets["DRUGBANK_API_KEY"]
    except Exception:
        return os.environ.get("DRUGBANK_API_KEY")

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Drug Discovery Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════

def inject_theme() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: white; }}
.stApp {{
    background: linear-gradient(135deg,{TEAL_DARK} 0%,#142850 30%,{TEAL_MID} 65%,#6A11CB 100%);
    color: white;
}}
.stat-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(0,212,255,.35);
    border-radius: 16px; padding: 22px 18px; text-align: center;
    backdrop-filter: blur(12px); transition: transform .2s, box-shadow .2s;
}}
.stat-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,212,255,.25); }}
.stat-card .icon {{ font-size: 2rem; margin-bottom: 6px; }}
.stat-card .val  {{ font-size: 1.9rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.stat-card .lbl  {{ color: #DCE9FF; font-size: .82rem; margin-top: 4px; }}
.metric-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(0,212,255,.35);
    border-radius: 14px; padding: 18px; text-align: center; backdrop-filter: blur(12px);
}}
.metric-card .val {{ font-size: 1.8rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.metric-card .lbl {{ color: #DCE9FF; font-size: .82rem; }}
.section-header {{
    border-left: 5px solid {TEAL_LIGHT}; padding-left: 12px; color: white;
    font-size: 1.2rem; font-weight: 700; margin: 25px 0 15px;
}}
.info-box {{
    background: rgba(0,212,255,.08); border: 1px solid rgba(0,212,255,.35);
    border-radius: 12px; padding: 16px; color: #EAF7FF;
}}
.disclaimer {{
    background: rgba(255,209,102,.08); border: 1px solid {GOLD};
    border-radius: 12px; padding: 14px; color: #FFE7A0;
}}
.ref-lig-card {{
    background: rgba(20,27,58,.9); border: 1px solid rgba(0,212,255,.25);
    border-radius: 14px; padding: 18px; color: white;
}}
.compound-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(123,97,255,.35);
    border-radius: 14px; padding: 18px; margin-bottom: 12px; color: white;
    backdrop-filter: blur(12px);
}}
.compound-card b {{ color: {TEAL_LIGHT}; }}
.roadmap-item {{
    background: rgba(20,27,58,.85); border-left: 4px solid {TEAL_LIGHT};
    border-radius: 10px; padding: 12px 16px; margin: 10px 0; color: white;
}}
.ai-bubble {{
    background: rgba(106,17,203,.25); border: 1px solid rgba(0,212,255,.3);
    border-radius: 14px 14px 14px 4px; padding: 14px 18px; color: white; margin-bottom: 10px;
}}
.user-bubble {{
    background: rgba(0,229,255,.12); border: 1px solid rgba(0,229,255,.3);
    border-radius: 14px 14px 4px 14px; padding: 14px 18px; color: white;
    margin-bottom: 10px; text-align: right;
}}
.progress-step {{
    background: rgba(20,27,58,.85); border: 1px solid rgba(0,212,255,.2);
    border-radius: 10px; padding: 10px 16px; margin: 6px 0; color: #DCE9FF; font-size: .9rem;
}}
.progress-step.done   {{ border-color: {GREEN_OK}; color: {GREEN_OK}; }}
.progress-step.active {{ border-color: {TEAL_LIGHT}; color: white; font-weight: 700; }}
.admet-pass {{ color: {GREEN_OK}; font-weight: 700; }}
.admet-warn {{ color: {GOLD}; font-weight: 700; }}
.admet-fail {{ color: {PINK}; font-weight: 700; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,{TEAL_DARK},{TEAL_MID} 120%);
}}
section[data-testid="stSidebar"] * {{ color: white !important; }}
section[data-testid="stSidebar"] .stTextInput input {{ color: black !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(20,27,58,.8); color: white;
    border-radius: 12px 12px 0 0; font-weight: 700; padding: 10px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg,#6A11CB,#00D4FF) !important; color: white !important;
}}
.stButton>button {{
    border-radius: 12px; border: none;
    background: linear-gradient(90deg,#6A11CB,#00D4FF);
    color: white; font-weight: 700; transition: .3s;
}}
.stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,212,255,.4); }}
.site-footer {{
    background: rgba(7,26,47,.95); border-top: 1px solid rgba(0,212,255,.25);
    border-radius: 16px; padding: 28px 32px; margin-top: 40px;
    text-align: center; color: #9DB4CC; font-size: .85rem;
}}
.site-footer b {{ color: {TEAL_LIGHT}; }}
.site-footer a  {{ color: {GOLD}; text-decoration: none; }}
.stTextInput input, .stSelectbox div, .stNumberInput input {{ border-radius: 10px !important; }}
.selectivity-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: .78rem; font-weight: 700; margin: 2px;
}}
</style>
""", unsafe_allow_html=True)

inject_theme()

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "history":           [],
    "results_df":        pd.DataFrame(),
    "protein_info":       None,
    "reports_count":     0,
    "ai_chat":           [],   # FIX #3: full multi-turn history
    "multi_target_results": {},
    "pubchem_admet_cache": {},
    "drugbank_cache":     {},
    "scaffold_cache":     {},
    "vina_available":     None,
    "pdf_ready":          False,  # FIX #2: track PDF generation separately
    "pdf_bytes":          None,
    "pdf_pdb_id":         None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def safe_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]

def score_col(df: pd.DataFrame) -> str:
    return SCORE_COL if SCORE_COL in df.columns else "docking_score"

def plotly_layout(extra: dict | None = None) -> dict:
    base = dict(
        plot_bgcolor="rgba(20,27,58,0.7)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    if extra:
        base.update(extra)
    return base

def metric_card(col, icon: str, value, label: str) -> None:
    col.markdown(
        f'<div class="stat-card">'
        f'<div class="icon">{icon}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _compound_count() -> int | str:
    try:
        return len(pd.read_csv(DATA_PATH))
    except Exception:
        return "N/A"

# ═══════════════════════════════════════════════════════════════════
# CLAUDE API  (FIX #1: proper API key + FIX #3: multi-turn)
# ═══════════════════════════════════════════════════════════════════

def _call_claude(system: str, messages: list[dict]) -> str:
    """
    Call Anthropic Messages API with full multi-turn history.
    API key from st.secrets or env var — never hardcoded.
    """
    api_key = _get_api_key()
    if not api_key:
        return ("⚠️ No API key found. Add ANTHROPIC_API_KEY to "
                ".streamlit/secrets.toml or set as environment variable.")

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "system": system,
        "messages": messages,   # full history passed every call (multi-turn)
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=45,
        )
        data = r.json()
        if "error" in data:
            return f"⚠️ API error: {data['error'].get('message', str(data['error']))}"
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
    except Exception as e:
        return f"⚠️ AI Assistant unavailable: {e}"

# ═══════════════════════════════════════════════════════════════════
# PUBCHEM LIVE ADMET  (NEW)
# ═══════════════════════════════════════════════════════════════════

def fetch_pubchem_admet(compound_name: str, cid: Optional[int] = None) -> dict:
    """
    Fetch live physicochemical + bioassay data from PubChem PUG-REST.
    Returns dict with ADMET-relevant properties.
    """
    cache_key = f"pc_{cid or compound_name}"
    if cache_key in st.session_state.pubchem_admet_cache:
        return st.session_state.pubchem_admet_cache[cache_key]

    result = {"source": "pubchem_live", "compound": compound_name}

    try:
        # If no CID, search by name
        if not cid:
            search_url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                          f"{requests.utils.quote(compound_name)}/cids/JSON")
            r = requests.get(search_url, timeout=8)
            if r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                cid = cids[0] if cids else None

        if cid:
            result["cid"] = cid
            # Fetch computed properties
            props = ("MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,"
                     "HBondAcceptorCount,RotatableBondCount,TPSA,Complexity,"
                     "HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,"
                     "BondStereoCount,CovalentUnitCount")
            prop_url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                        f"{cid}/property/{props}/JSON")
            pr = requests.get(prop_url, timeout=8)
            if pr.status_code == 200:
                props_data = pr.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                result.update({
                    "molecular_formula":  props_data.get("MolecularFormula", "—"),
                    "molecular_weight":   props_data.get("MolecularWeight", "—"),
                    "xlogp":              props_data.get("XLogP", "—"),
                    "h_donors":           props_data.get("HBondDonorCount", "—"),
                    "h_acceptors":        props_data.get("HBondAcceptorCount", "—"),
                    "rotatable_bonds":    props_data.get("RotatableBondCount", "—"),
                    "tpsa":               props_data.get("TPSA", "—"),
                    "complexity":         props_data.get("Complexity", "—"),
                    "heavy_atoms":        props_data.get("HeavyAtomCount", "—"),
                })

            # Fetch bioassay summary
            assay_url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                         f"{cid}/assaysummary/JSON")
            ar = requests.get(assay_url, timeout=8)
            if ar.status_code == 200:
                assays = ar.json().get("Table", {}).get("Row", [])
                active = sum(
                    1 for row in assays
                    if any(
                        cell.get("Value", "") == "Active"
                        for cell in (row.get("Cell") or [])
                    )
                )
                result["bioassay_active_count"] = active
                result["bioassay_total_count"]  = len(assays)

            # Validated ADMET flags
            result["admet_flags"] = _compute_admet_flags(result)

    except Exception as e:
        result["error"] = str(e)

    st.session_state.pubchem_admet_cache[cache_key] = result
    return result


def _compute_admet_flags(props: dict) -> dict:
    """
    Compute validated ADMET flags from physicochemical data.
    Returns dict of property → (status, note).
    """
    flags = {}

    def _safe(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    mw   = _safe(props.get("molecular_weight"))
    logp = _safe(props.get("xlogp"))
    hbd  = _safe(props.get("h_donors"))
    hba  = _safe(props.get("h_acceptors"))
    rb   = _safe(props.get("rotatable_bonds"))
    tpsa = _safe(props.get("tpsa"))

    # Lipinski Ro5
    violations = sum([
        mw   is not None and mw   > 500,
        logp is not None and logp > 5,
        hbd  is not None and hbd  > 5,
        hba  is not None and hba  > 10,
    ])
    flags["Lipinski Ro5"] = (
        ("PASS", f"{violations} violation(s)") if violations == 0
        else ("WARN" if violations == 1 else "FAIL", f"{violations} violation(s)")
    )

    # Oral absorption (Veber rules: TPSA ≤ 140, RotBonds ≤ 10)
    if tpsa is not None and rb is not None:
        if tpsa <= 140 and rb <= 10:
            flags["Oral Absorption"] = ("PASS", f"TPSA={tpsa}, RotBonds={rb}")
        else:
            flags["Oral Absorption"] = ("WARN", f"TPSA={tpsa}, RotBonds={rb}")

    # BBB penetration (crude: MW < 450, LogP 1-3, TPSA < 90)
    if mw is not None and logp is not None:
        bbb_ok = mw < 450 and 1 <= logp <= 3 and (tpsa is None or tpsa < 90)
        flags["BBB Penetration"] = (
            "PASS" if bbb_ok else "WARN",
            f"MW={mw}, LogP={logp}"
        )

    # hERG liability (LogP > 3.7 and MW > 300 → flag)
    if logp is not None and mw is not None:
        herg_risk = logp > 3.7 and mw > 300
        flags["hERG Liability"] = (
            "WARN" if herg_risk else "PASS",
            "High lipophilicity risk" if herg_risk else "Low risk"
        )

    # Drug-likeness score (0–5)
    score = sum([
        mw   is not None and 150 <= mw   <= 500,
        logp is not None and -1  <= logp <= 5,
        hbd  is not None and hbd  <= 5,
        hba  is not None and hba  <= 10,
        rb   is not None and rb   <= 10,
    ])
    flags["Drug-Likeness Score"] = (
        "PASS" if score >= 4 else ("WARN" if score >= 2 else "FAIL"),
        f"{score}/5 criteria met"
    )

    return flags

# ═══════════════════════════════════════════════════════════════════
# DRUGBANK API  (NEW)
# ═══════════════════════════════════════════════════════════════════

def fetch_drugbank_info(compound_name: str, drugbank_id: Optional[str] = None) -> dict:
    """
    Fetch drug information from DrugBank API.
    Falls back to PubChem + curated data when no API key.
    """
    cache_key = f"db_{drugbank_id or compound_name}"
    if cache_key in st.session_state.drugbank_cache:
        return st.session_state.drugbank_cache[cache_key]

    result = {
        "compound": compound_name,
        "drugbank_id": drugbank_id,
        "source": "unavailable",
    }

    db_key = _get_drugbank_key()

    if db_key and drugbank_id:
        try:
            headers = {
                "Authorization": f"Bearer {db_key}",
                "Content-Type":  "application/json",
            }
            r = requests.get(
                f"https://api.drugbank.com/v1/drugs/{drugbank_id}",
                headers=headers, timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                result.update({
                    "source":         "drugbank_live",
                    "name":           data.get("name", "—"),
                    "description":    data.get("description", "—"),
                    "mechanism":      data.get("mechanism_of_action", "—"),
                    "indication":     data.get("indication", "—"),
                    "pharmacodynamics": data.get("pharmacodynamics", "—"),
                    "half_life":      data.get("half_life", "—"),
                    "protein_binding": data.get("protein_binding", "—"),
                    "metabolism":     data.get("metabolism", "—"),
                    "toxicity":       data.get("toxicity", "—"),
                    "categories":     [c.get("name", "") for c in data.get("categories", [])],
                    "interactions":   [
                        {
                            "drug":        i.get("drug", {}).get("name", "—"),
                            "description": i.get("description", "—"),
                            "severity":    i.get("severity", "—"),
                        }
                        for i in data.get("drug_interactions", [])[:10]
                    ],
                    "targets": [
                        {
                            "name":   t.get("name", "—"),
                            "gene":   t.get("gene_name", "—"),
                            "action": t.get("actions", ["—"])[0] if t.get("actions") else "—",
                        }
                        for t in data.get("targets", [])[:8]
                    ],
                })
        except Exception as e:
            result["error"] = str(e)

    # Fallback: curated lookup for well-known drugs
    if result["source"] == "unavailable":
        result.update(_drugbank_curated_fallback(compound_name))

    st.session_state.drugbank_cache[cache_key] = result
    return result


def _drugbank_curated_fallback(compound_name: str) -> dict:
    """Curated offline data for common drugs used in virtual screening."""
    CURATED = {
        "oseltamivir": {
            "source": "curated",
            "mechanism": "Competitive inhibitor of influenza neuraminidase, preventing viral release from infected cells.",
            "indication": "Treatment and prophylaxis of influenza A and B.",
            "half_life": "6–10 hours (active metabolite oseltamivir carboxylate)",
            "protein_binding": "42% (oseltamivir carboxylate: 3%)",
            "metabolism": "Hepatic ester hydrolysis to active carboxylate form.",
            "categories": ["Antiviral", "Neuraminidase Inhibitor"],
        },
        "lopinavir": {
            "source": "curated",
            "mechanism": "HIV-1 protease inhibitor preventing viral polyprotein processing.",
            "indication": "HIV-1 infection (in combination with ritonavir).",
            "half_life": "5–6 hours",
            "protein_binding": "98–99%",
            "metabolism": "Hepatic CYP3A4.",
            "categories": ["Antiretroviral", "Protease Inhibitor"],
        },
        "ibuprofen": {
            "source": "curated",
            "mechanism": "Non-selective COX-1/COX-2 inhibitor reducing prostaglandin synthesis.",
            "indication": "Pain, fever, inflammation.",
            "half_life": "2 hours",
            "protein_binding": "99%",
            "metabolism": "Hepatic CYP2C9.",
            "categories": ["NSAID", "Analgesic", "Anti-inflammatory"],
        },
        "remdesivir": {
            "source": "curated",
            "mechanism": "Prodrug adenosine nucleotide analogue inhibiting viral RNA-dependent RNA polymerase.",
            "indication": "COVID-19 treatment (hospitalised patients).",
            "half_life": "~1 hour (parent); active metabolite ~27 hours",
            "protein_binding": "88–93.6%",
            "metabolism": "Intracellular hydrolysis to active triphosphate form.",
            "categories": ["Antiviral", "Nucleotide Analogue"],
        },
    }
    name_lower = compound_name.lower()
    for key, data in CURATED.items():
        if key in name_lower:
            return {**data, "compound": compound_name}
    return {"source": "not_found", "note": "No curated data. Add DRUGBANK_API_KEY for live data."}

# ═══════════════════════════════════════════════════════════════════
# SCAFFOLD CLUSTERING  (NEW — RDKit)
# ═══════════════════════════════════════════════════════════════════

def compute_scaffolds(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Bemis-Murcko scaffolds for all compounds with SMILES.
    Returns dataframe with scaffold column.
    """
    if not RDKIT_AVAILABLE:
        return compounds_df.copy()

    cache_key = hashlib.md5(str(compounds_df["compound_name"].tolist()).encode()).hexdigest()
    if cache_key in st.session_state.scaffold_cache:
        return st.session_state.scaffold_cache[cache_key]

    rows = []
    for _, row in compounds_df.iterrows():
        smiles = row.get("smiles", "")
        scaffold = "No SMILES"
        scaffold_smiles = ""
        if smiles and str(smiles) not in ("nan", "None", ""):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(core)
                    scaffold = scaffold_smiles if scaffold_smiles else "No scaffold"
            except Exception:
                scaffold = "Error"
        rows.append({**row.to_dict(), "scaffold": scaffold, "scaffold_smiles": scaffold_smiles})

    result = pd.DataFrame(rows)
    st.session_state.scaffold_cache[cache_key] = result
    return result


def render_scaffold_clustering(compounds_df: pd.DataFrame) -> None:
    """Render Bemis-Murcko scaffold clustering analysis."""
    st.markdown('<div class="section-header">🧪 Scaffold Clustering</div>',
                unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.warning("RDKit not installed. Run `pip install rdkit` to enable scaffold clustering.")
        st.info("Showing compound grouping by molecular weight as fallback.")
        # Fallback: bin by MW
        if "molecular_weight" in compounds_df.columns:
            df = compounds_df.copy()
            df["mw_bin"] = pd.cut(
                df["molecular_weight"].astype(float),
                bins=[0, 300, 400, 500, 700, 9999],
                labels=["<300", "300-400", "400-500", "500-700", ">700"],
            )
            fig = px.histogram(
                df, x="mw_bin", color="mw_bin",
                title="Compounds by Molecular Weight Range",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_layout(**plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        return

    with st.spinner("Computing Bemis-Murcko scaffolds…"):
        df_scaffolds = compute_scaffolds(compounds_df)

    if "scaffold" not in df_scaffolds.columns:
        st.info("No SMILES data available for scaffold computation.")
        return

    # Scaffold frequency
    scaffold_counts = (
        df_scaffolds[df_scaffolds["scaffold"] != "No SMILES"]
        .groupby("scaffold")["compound_name"]
        .apply(list)
        .reset_index()
    )
    scaffold_counts["count"] = scaffold_counts["compound_name"].apply(len)
    scaffold_counts = scaffold_counts.sort_values("count", ascending=False)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric("Unique Scaffolds", len(scaffold_counts))
        st.metric("Singleton Scaffolds", int((scaffold_counts["count"] == 1).sum()))
    with sc2:
        st.metric("Scaffold Diversity", f"{len(scaffold_counts)/max(len(df_scaffolds),1)*100:.0f}%")
        top_scaffold = scaffold_counts.iloc[0]["scaffold"] if len(scaffold_counts) else "—"
        st.metric("Most Common Scaffold", top_scaffold[:30] + "…" if len(top_scaffold) > 30 else top_scaffold)

    # Treemap by scaffold
    if len(scaffold_counts) > 0:
        exploded = scaffold_counts.explode("compound_name")
        fig_tree = px.treemap(
            exploded,
            path=["scaffold", "compound_name"],
            title="Scaffold Hierarchy Treemap",
            color="count",
            color_continuous_scale="Viridis",
        )
        fig_tree.update_layout(**plotly_layout())
        st.plotly_chart(fig_tree, use_container_width=True)

    # Scaffold table
    with st.expander("View scaffold assignments"):
        st.dataframe(
            df_scaffolds[safe_cols(df_scaffolds, [
                "compound_name", "molecular_formula", "molecular_weight", "scaffold"
            ])].rename(columns={
                "compound_name": "Compound",
                "molecular_formula": "Formula",
                "molecular_weight": "MW",
                "scaffold": "Bemis-Murcko Scaffold",
            }),
            use_container_width=True, hide_index=True,
        )

    # Hierarchical clustering dendrogram (if scipy available)
    if SCIPY_AVAILABLE and RDKIT_AVAILABLE:
        num_cols = safe_cols(compounds_df, [
            "molecular_weight", "logp", "h_donors", "h_acceptors", "rotatable_bonds"
        ])
        if len(num_cols) >= 3 and len(compounds_df) >= 3:
            try:
                X = compounds_df[num_cols].fillna(0).values
                Z = linkage(X, method="ward")
                labels = compounds_df["compound_name"].tolist()

                fig_dend = go.Figure()
                dend = dendrogram(Z, labels=labels, no_plot=True)
                for i, (xs, ys) in enumerate(zip(dend["icoord"], dend["dcoord"])):
                    fig_dend.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line=dict(color=TEAL_LIGHT, width=1.5),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig_dend.update_layout(
                    title="Hierarchical Clustering Dendrogram (Ward, Physicochemical Properties)",
                    xaxis=dict(tickvals=dend["leaves"], ticktext=labels, tickangle=-45),
                    **plotly_layout({"height": 400}),
                )
                st.plotly_chart(fig_dend, use_container_width=True)
            except Exception as e:
                st.caption(f"Dendrogram unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════
# MULTI-TARGET SCREENING  (NEW)
# ═══════════════════════════════════════════════════════════════════

POPULAR_TARGETS = {
    "6LU7": "SARS-CoV-2 Main Protease",
    "2HU4": "Influenza Neuraminidase",
    "1HVR": "HIV-1 Protease",
    "3POZ": "PI3K Kinase",
    "1EQG": "COX-1 + Ibuprofen",
    "1ATP": "cAMP-Dependent Protein Kinase",
    "2V0Z": "SARS-CoV-2 Spike RBD",
    "4DKL": "Beta-Secretase 1 (BACE1)",
}


def render_multi_target_tab() -> None:
    st.markdown('<div class="section-header">🗂️ Multi-Target Selectivity Screening</div>',
                unsafe_allow_html=True)
    st.write("Screen your compound library against multiple protein targets simultaneously "
             "and generate a selectivity heatmap.")

    compounds_df = screening.load_compound_data(DATA_PATH)
    all_compounds = compounds_df["compound_name"].tolist()

    col1, col2 = st.columns([2, 3])
    with col1:
        # Target selection
        st.markdown("**Select Protein Targets**")
        selected_targets = st.multiselect(
            "PDB IDs to screen against",
            options=list(POPULAR_TARGETS.keys()),
            default=["6LU7", "2HU4", "1HVR"],
            format_func=lambda x: f"{x} — {POPULAR_TARGETS[x]}",
        )
        custom_target = st.text_input(
            "Add custom PDB ID", max_chars=4, placeholder="e.g. 3ERT"
        ).strip().upper()
        if custom_target and custom_target not in selected_targets:
            selected_targets = selected_targets + [custom_target]

        # Compound selection
        st.markdown("**Select Compounds**")
        selected_compounds = st.multiselect(
            "Compounds to screen",
            options=all_compounds,
            default=all_compounds[:6],
        )

    with col2:
        st.markdown("**Targets Overview**")
        for pdb, name in POPULAR_TARGETS.items():
            if pdb in selected_targets:
                st.markdown(
                    f'<div class="compound-card" style="padding:10px">'
                    f'<b>{pdb}</b> — {name}</div>',
                    unsafe_allow_html=True,
                )

    if not selected_targets or not selected_compounds:
        st.info("Select at least one target and one compound to begin.")
        return

    if st.button("🚀 Run Multi-Target Screening", type="primary", use_container_width=True):
        progress = st.progress(0)
        status   = st.empty()
        results  = {}

        for i, pdb_id in enumerate(selected_targets):
            status.info(f"⏳ Screening against {pdb_id} ({i+1}/{len(selected_targets)})…")
            prot_info = protein.get_protein_info(pdb_id)
            raw = screening.simulate_virtual_screening(
                compounds_df, pdb_id, selected_compounds, protein_info=prot_info
            )
            ranked = ranking.rank_compounds(raw)
            sc = score_col(ranked)
            results[pdb_id] = {c: ranked[ranked["compound_name"] == c][sc].values[0]
                               for c in selected_compounds
                               if c in ranked["compound_name"].values}
            progress.progress(int((i + 1) / len(selected_targets) * 100))

        st.session_state.multi_target_results = results
        status.success("✅ Multi-target screening complete!")
        progress.empty()

    if st.session_state.multi_target_results:
        results = st.session_state.multi_target_results
        st.markdown("---")
        st.markdown("#### 🗺️ Selectivity Heatmap")

        # Build matrix
        pdb_ids = list(results.keys())
        compounds_list = list(next(iter(results.values())).keys()) if results else []
        matrix = pd.DataFrame(
            {pdb: [results[pdb].get(c, None) for c in compounds_list] for pdb in pdb_ids},
            index=compounds_list,
        )

        fig_heat = go.Figure(go.Heatmap(
            z=matrix.values,
            x=[f"{p}<br>{POPULAR_TARGETS.get(p, '')}" for p in pdb_ids],
            y=compounds_list,
            colorscale="RdYlGn_r",
            text=matrix.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Score<br>(kcal/mol)"),
        ))
        fig_heat.update_layout(
            title="Multi-Target Selectivity Heatmap (lower = stronger binding)",
            height=max(400, len(compounds_list) * 40),
            **plotly_layout(),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Best target per compound
        st.markdown("#### 🏆 Selectivity Summary")
        summary_rows = []
        for compound in compounds_list:
            scores = {pdb: results[pdb].get(compound) for pdb in pdb_ids if results[pdb].get(compound) is not None}
            if scores:
                best_target = min(scores, key=scores.get)
                worst_target = max(scores, key=scores.get)
                selectivity_ratio = scores[worst_target] - scores[best_target]
                summary_rows.append({
                    "Compound":       compound,
                    "Best Target":    f"{best_target} ({POPULAR_TARGETS.get(best_target,'')})",
                    "Best Score":    f"{scores[best_target]:.2f}",
                    "Selectivity Δ":   f"{selectivity_ratio:.2f}",
                })

        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True, hide_index=True,
            )

        # Radar: top compound across targets
        if len(compounds_list) >= 1 and len(pdb_ids) >= 3:
            top_compound = min(
                compounds_list,
                key=lambda c: min((results[p].get(c, 0) for p in pdb_ids), default=0),
            )
            radar_vals = [abs(results[p].get(top_compound, 0)) for p in pdb_ids]
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=[f"{p}" for p in pdb_ids] + [pdb_ids[0]],
                fill="toself",
                name=top_compound,
                line_color=TEAL_LIGHT,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=f"Target Coverage Radar — {top_compound}",
                **plotly_layout(),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# AUTODOCK VINA INTEGRATION  (NEW)
# ═══════════════════════════════════════════════════════════════════

def check_vina_available() -> bool:
    """Check if AutoDock Vina binary is in PATH."""
    if st.session_state.vina_available is not None:
        return st.session_state.vina_available
    try:
        result = subprocess.run(
            ["vina", "--version"], capture_output=True, text=True, timeout=5
        )
        available = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        available = False
    st.session_state.vina_available = available
    return available


def render_vina_tab() -> None:
    st.markdown('<div class="section-header">🔬 AutoDock Vina Integration</div>',
                unsafe_allow_html=True)

    vina_ok = check_vina_available()

    if vina_ok:
        st.success("✅ AutoDock Vina detected in PATH — real docking available!")
    else:
        st.warning(
            "⚠️ AutoDock Vina not found in PATH. Showing setup instructions and "
            "simulation mode. Install Vina to enable real docking."
        )

    with st.expander("📦 AutoDock Vina Installation Guide", expanded=not vina_ok):
        st.markdown("""
### Installing AutoDock Vina

**Option 1 — Conda (recommended)**
```bash
conda install -c conda-forge autodock-vina
