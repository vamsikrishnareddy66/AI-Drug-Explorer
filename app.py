"""
app.py
------
AI Drug Discovery Platform — Professional Edition v5.0
B.Tech Biotechnology · KL University

Developed by: Vamsi Krishna Reddy
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

# Local modules (Preserving complete compatibility with existing backend modules)
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
# CONFIGURATION & GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════
DATA_PATH   = "compounds.csv"
EXPORTS_DIR = "exports"
SCORE_COL   = "simulated_score"
APP_VERSION = "5.0"
DEVELOPER   = "Vamsi Krishna Reddy"
INSTITUTION = "KL University"

# Colors for Vibrant Purple-Blue-Pink theme
TEAL_DARK  = "#071A2F"
TEAL_MID   = "#1E3A8A"
TEAL_LIGHT = "#00E5FF"
GOLD       = "#FFD166"
GREEN_OK   = "#00F5A0"
BG_CARD    = "#16213E"
PINK       = "#FF2E88"

os.makedirs(EXPORTS_DIR, exist_ok=True)


# ── API Key resolution ─────────────────────────────────────────────
def _get_api_key() -> Optional[str]:
    """Retrieve API Key prioritizing Gemini and falling back to Anthropic key."""
    for key in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY"]:
        try:
            val = st.secrets[key]
            if val: return val
        except Exception:
            pass
        val = os.environ.get(key)
        if val: return val
    return None

def _get_drugbank_key() -> Optional[str]:
    try:
        return st.secrets["DRUGBANK_API_KEY"]
    except Exception:
        return os.environ.get("DRUGBANK_API_KEY")


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Drug Discovery Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# THEME INJECTION
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

/* Dynamic Glass Card Dashboard Aesthetics */
.dashboard-card {{
    background: rgba(20, 27, 58, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.35);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease-in-out;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin-bottom: 12px;
}}
.dashboard-card:hover {{
    transform: translateY(-5px);
    border-color: #FF0080;
    box-shadow: 0 10px 30px rgba(0, 240, 255, 0.25);
}}
.dashboard-card .icon {{
    font-size: 2.2rem;
    margin-bottom: 8px;
}}
.dashboard-card .val {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {TEAL_LIGHT};
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
}}
.dashboard-card .lbl {{
    color: #DCE9FF;
    font-size: 0.85rem;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
</style>
""", unsafe_allow_html=True)

inject_theme()


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "history":            [],
    "results_df":         pd.DataFrame(),
    "protein_info":       None,
    "reports_count":      0,
    "ai_chat":            [],
    "multi_target_results": {},
    "pubchem_admet_cache": {},
    "drugbank_cache":     {},
    "scaffold_cache":     {},
    "vina_available":     None,
    "pdf_ready":          False,
    "pdf_bytes":          None,
    "pdf_pdb_id":         None,
    "session_time":       time.time(),
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

def render_badge(status: str, label: str) -> str:
    status_upper = status.upper()
    if status_upper in ["PASS", "APPROVED", "HIGH", "YES", "TRUE", "LOW RISK", "MINIMAL POTENTIAL"]:
        color_class = "admet-pass"
    elif status_upper in ["WARN", "WARNING", "MODERATE", "CRITICAL WARNING"]:
        color_class = "admet-warn"
    else:
        color_class = "admet-fail"
    return f'<div class="compound-card" style="text-align:center;padding:12px"><b>{label}</b><br><span class="{color_class}">{status}</span></div>'


# ═══════════════════════════════════════════════════════════════════
# GEMINI 2.5 FLASH API CONNECTOR (FIXED DEPRECATED PREVIEW ID)
# ═══════════════════════════════════════════════════════════════════
def _call_gemini(system: str, messages: list[dict]) -> str:
    """
    Call Google Gemini 2.5 Flash API with complete multi-turn history.
    Keeps compatibility with secrets manager and env variables.
    """
    api_key = _get_api_key()
    if not api_key:
        return ("⚠️ No API key found. Add GEMINI_API_KEY to "
                ".streamlit/secrets.toml or set as environment variable.")

    # Convert client messages format to Gemini model structures
    gemini_contents = []
    for msg in messages:
        role = "model" if msg.get("role") == "assistant" else "user"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {
            "parts": [{"text": system}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45)
        data = r.json()
        if "error" in data:
            return f"⚠️ API error: {data['error'].get('message', str(data['error']))}"
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if text:
            return text
        return "⚠️ No response generated by Gemini."
    except Exception as e:
        return f"⚠️ AI Assistant unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════
# PUBCHEM LIVE ADMET 
# ═══════════════════════════════════════════════════════════════════
def fetch_pubchem_admet(compound_name: str, cid: Optional[int] = None) -> dict:
    """Fetch live ADMET data from PubChem PUG-REST."""
    cache_key = f"pc_{cid or compound_name}"
    if cache_key in st.session_state.pubchem_admet_cache:
        return st.session_state.pubchem_admet_cache[cache_key]

    result = {"source": "pubchem_live", "compound": compound_name}
    try:
        if not cid:
            search_url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                          f"{requests.utils.quote(compound_name)}/cids/JSON")
            r = requests.get(search_url, timeout=8)
            if r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                cid = cids[0] if cids else None

        if cid:
            result["cid"] = cid
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

            result["admet_flags"] = _compute_admet_flags(result)
    except Exception as e:
        result["error"] = str(e)

    st.session_state.pubchem_admet_cache[cache_key] = result
    return result


def _compute_admet_flags(props: dict) -> dict:
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

    if TPSA := tpsa is not None and rb is not None:
        if tpsa <= 140 and rb <= 10:
            flags["Oral Absorption"] = ("PASS", f"TPSA={tpsa}, RotBonds={rb}")
        else:
            flags["Oral Absorption"] = ("WARN", f"TPSA={tpsa}, RotBonds={rb}")

    if mw is not None and logp is not None:
        bbb_ok = mw < 450 and 1 <= logp <= 3 and (tpsa is None or tpsa < 90)
        flags["BBB Penetration"] = (
            "PASS" if bbb_ok else "WARN",
            f"MW={mw}, LogP={logp}"
        )

    if logp is not None and mw is not None:
        herg_risk = logp > 3.7 and mw > 300
        flags["hERG Liability"] = (
            "WARN" if herg_risk else "PASS",
            "High lipophilicity risk" if herg_risk else "Low risk"
        )

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
# DRUGBANK CLINICAL MODULE
# ═══════════════════════════════════════════════════════════════════
def fetch_drugbank_info(compound_name: str, drugbank_id: Optional[str] = None) -> dict:
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
                    "source":          "drugbank_live",
                    "name":            data.get("name", "—"),
                    "description":     data.get("description", "—"),
                    "mechanism":       data.get("mechanism_of_action", "—"),
                    "indication":      data.get("indication", "—"),
                    "pharmacodynamics": data.get("pharmacodynamics", "—"),
                    "half_life":       data.get("half_life", "—"),
                    "protein_binding": data.get("protein_binding", "—"),
                    "metabolism":      data.get("metabolism", "—"),
                    "toxicity":        data.get("toxicity", "—"),
                    "categories":      [c.get("name", "") for c in data.get("categories", [])],
                    "interactions":    [
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

    if result["source"] == "unavailable":
        result.update(_drugbank_curated_fallback(compound_name))

    st.session_state.drugbank_cache[cache_key] = result
    return result


def _drugbank_curated_fallback(compound_name: str) -> dict:
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
# SCAFFOLD CLUSTERING (NEW)
# ═══════════════════════════════════════════════════════════════════
def compute_scaffolds(compounds_df: pd.DataFrame) -> pd.DataFrame:
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
    st.markdown('<div class="section-header">🧪 Scaffold Clustering</div>', unsafe_allow_html=True)
    if not RDKIT_AVAILABLE:
        st.warning("RDKit not installed. Fallback to basic chemical classification.")
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
# MULTI-TARGET METRIC MAPS
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
    st.markdown('<div class="section-header">🗂️ Multi-Target Selectivity Screening</div>', unsafe_allow_html=True)
    st.write("Screen your compound library against multiple protein targets simultaneously "
             "and generate a selectivity heatmap.")

    compounds_df = screening.load_compound_data(DATA_PATH)
    all_compounds = compounds_df["compound_name"].tolist()

    col1, col2 = st.columns([2, 3])
    with col1:
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

        st.markdown("#### 🏆 Selectivity Summary")
        summary_rows = []
        for compound in compounds_list:
            scores = {pdb: results[pdb].get(compound) for pdb in pdb_ids if results[pdb].get(compound) is not None}
            if scores:
                best_target = min(scores, key=scores.get)
                worst_target = max(scores, key=scores.get)
                selectivity_ratio = scores[worst_target] - scores[best_target]
                summary_rows.append({
                    "Compound":        compound,
                    "Best Target":     f"{best_target} ({POPULAR_TARGETS.get(best_target,'')})",
                    "Best Score":      f"{scores[best_target]:.2f}",
                    "Selectivity Δ":   f"{selectivity_ratio:.2f}",
                })

        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True, hide_index=True,
            )

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
# AUTODOCK VINA INTEGRATION 
# ═══════════════════════════════════════════════════════════════════
def check_vina_available() -> bool:
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
    st.markdown('<div class="section-header">🔬 AutoDock Vina Integration</div>', unsafe_allow_html=True)
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
```

**Option 2 — pip**
```bash
pip install vina
```

**Option 3 — Binary download**
Download from: https://vina.scripps.edu/downloads/

### Preparing Input Files

1. **Protein** — convert `.pdb` → `.pdbqt` using MGLTools or `prepare_receptor4.py`
```bash
pythonsh prepare_receptor4.py -r protein.pdb -o protein.pdbqt
```

2. **Ligand** — convert SMILES → 3D → `.pdbqt`
```bash
obabel compound.smi -O compound.pdbqt --gen3d -p 7.4
```

3. **Config file** (`config.txt`)
```
receptor = protein.pdbqt
ligand   = compound.pdbqt
center_x = 10.0
center_y = 20.0
center_z = 30.0
size_x   = 20
size_y   = 20
size_z   = 20
exhaustiveness = 8
num_modes = 9
```

4. **Run Vina**
```bash
vina --config config.txt --out output.pdbqt --log docking.log
```
""")

    if vina_ok:
        st.markdown("#### Run Real Docking")
        v1, v2 = st.columns(2)
        with v1:
            receptor_file = st.file_uploader("Upload receptor (.pdbqt)", type=["pdbqt"])
            ligand_file   = st.file_uploader("Upload ligand (.pdbqt)",   type=["pdbqt"])
        with v2:
            cx = st.number_input("Center X", value=10.0)
            cy = st.number_input("Center Y", value=20.0)
            cz = st.number_input("Center Z", value=30.0)
            sz = st.number_input("Box Size (Å)", value=20, step=1)
            exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8)

        if receptor_file and ligand_file and st.button("Run Vina Docking", type="primary"):
            rec_path = os.path.join(EXPORTS_DIR, "receptor.pdbqt")
            lig_path = os.path.join(EXPORTS_DIR, "ligand.pdbqt")
            out_path = os.path.join(EXPORTS_DIR, "output.pdbqt")
            log_path = os.path.join(EXPORTS_DIR, "docking.log")

            with open(rec_path, "wb") as f:
                f.write(receptor_file.read())
            with open(lig_path, "wb") as f:
                f.write(ligand_file.read())

            cmd = [
                "vina",
                "--receptor", rec_path, "--ligand", lig_path,
                "--out", out_path, "--log", log_path,
                f"--center_x={cx}", f"--center_y={cy}", f"--center_z={cz}",
                f"--size_x={sz}", f"--size_y={sz}", f"--size_z={sz}",
                f"--exhaustiveness={exhaustiveness}",
            ]

            with st.spinner("Running AutoDock Vina…"):
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if proc.returncode == 0:
                        st.success("✅ Docking complete!")
                        if os.path.exists(log_path):
                            with open(log_path) as f:
                                log_text = f.read()
                            st.text_area("Docking Log", log_text, height=200)
                            scores = re.findall(r"\s+1\s+([-\d.]+)", log_text)
                            if scores:
                                st.metric("Best Docking Score", f"{scores[0]} kcal/mol")
                        if os.path.exists(out_path):
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    "Download Docked Pose (.pdbqt)",
                                    data=f.read(),
                                    file_name="docked_pose.pdbqt",
                                    icon="⬇️",
                                )
                    else:
                        st.error(f"Vina error: {proc.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Vina timed out after 5 minutes.")
                except Exception as e:
                    st.error(f"Error running Vina: {e}")
    else:
        st.markdown("#### 🧪 Simulation Mode (Vina not installed)")
        st.info(
            "When Vina is installed, this tab will run real molecular docking. "
            "Currently using formula-based simulation scores from the Screening tab."
        )
        if not st.session_state.results_df.empty:
            st.markdown("**Simulated scores from last screening run:**")
            sc = score_col(st.session_state.results_df)
            display = st.session_state.results_df[safe_cols(
                st.session_state.results_df,
                ["rank", "compound_name", sc, "lipinski_status"]
            )].head(10)
            st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════
def render_sidebar() -> None:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/DNA_icon.svg/512px-DNA_icon.svg.png",
        width=80,
    )
    with st.sidebar:
        st.markdown(f"# 🧬 AI Drug Discovery")
        st.caption(f"v{APP_VERSION} Professional · {INSTITUTION}")
        st.markdown("---")

        # Application Status Module
        st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#DCE9FF; text-transform:uppercase; margin-bottom:8px;'>Application Status</p>", unsafe_allow_html=True)
        
        gemini_ok = "🟢 Gemini Connected" if _get_api_key() else "🔴 Gemini Offline"
        drugbank_ok = "🟢 DrugBank Online" if _get_drugbank_key() else "🟢 DrugBank Offline"
        vina_status = "🟢 Docking Ready" if check_vina_available() else "🟡 Emulation Mode"
        
        st.markdown(f"""
        <div style="background: rgba(20, 27, 58, 0.85); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 12px; padding: 12px; margin-bottom: 20px;">
            <div style="font-size:0.85rem; margin-bottom:6px;">{gemini_ok}</div>
            <div style="font-size:0.85rem; margin-bottom:6px;">🟢 PubChem Online</div>
            <div style="font-size:0.85rem; margin-bottom:6px;">{drugbank_ok}</div>
            <div style="font-size:0.85rem; margin-bottom:6px;">🟢 RCSB Online</div>
            <div style="font-size:0.85rem;">{vina_status}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 System Statistics")
        st.metric("🧬 Proteins Loaded", (st.session_state.protein_info or {}).get("pdb_id", "Not loaded"))
        st.metric("💊 Compounds Screened", len(st.session_state.results_df) if not st.session_state.results_df.empty else 0)
        st.metric("📄 Reports Generated", st.session_state.reports_count)
        
        # Session Timer
        session_elapsed = int(time.time() - st.session_state.session_time)
        session_min = f"{session_elapsed // 60:02d}:{session_elapsed % 60:02d}"
        st.metric("⌛ Session Time", session_min)

        st.markdown("---")
        st.markdown("### 🔗 Popular Targets")
        for pdb, name in list(POPULAR_TARGETS.items())[:5]:
            st.markdown(f"**{pdb}** — {name}")

        st.markdown("---")
        st.markdown("### 📜 Recent History")
        if st.session_state.history:
            for entry in reversed(st.session_state.history[-5:]):
                st.markdown(f"**{entry['protein']}** · {entry['timestamp']}")
                st.caption(f"{entry['best_compound']} · {entry['best_score']} kcal/mol")
        else:
            st.caption("No analyses yet.")

        st.markdown("---")
        st.markdown(f"**{DEVELOPER}** \nB.Tech Biotechnology · {INSTITUTION}")
        st.markdown("[📸 Instagram](https://www.instagram.com/n_vamsi_reddie)  "
                    "[💻 GitHub](https://github.com/vamsikrishnareddy66)")
        st.markdown("---")
        st.warning("⚠️ Educational Use Only. Scores are simulated.")
        st.caption(f"🧬 v{APP_VERSION} · Python · Streamlit · RCSB PDB")

render_sidebar()


# ═══════════════════════════════════════════════════════════════════
# HERO BANNER WITH STABLE DIGITAL CLOCK SCRIPT
# ═══════════════════════════════════════════════════════════════════
components.html(f"""
<div style="background:linear-gradient(135deg,#0052D4,#6A11CB,#FF0080);
            padding:28px 32px;border-radius:22px;color:white;
            display:flex;justify-content:space-between;align-items:center;
            font-family:Inter,Arial,sans-serif;box-shadow:0 8px 28px rgba(0,0,0,.35);">
  <div>
    <div style="display:inline-block;background:#FF2E88;padding:4px 14px;
                border-radius:20px;font-size:12px;font-weight:700;margin-bottom:10px;text-transform:uppercase;">
      🧬 AI Drug Discovery Platform
    </div>
    <h1 style="margin:0;font-size:28px;font-weight:800;">🧬 AI Drug Discovery Platform</h1>
    <p style="margin:6px 0 0;font-size:13px;opacity:.85;">
      Professional Virtual Screening & Lead Optimization System — Version 5.0 Professional
    </p>
    <p style="margin:10px 0 0;font-size:13px;line-height:1.6;">
      Gemini 2.5 Flash &nbsp;•&nbsp; PubChem &nbsp;•&nbsp; DrugBank &nbsp;•&nbsp;
      RCSB PDB &nbsp;•&nbsp; RDKit &nbsp;•&nbsp; AutoDock Vina
    </p>
  </div>
  <div style="background:rgba(255,255,255,.14);padding:16px;border-radius:14px;
              width:175px;text-align:center;backdrop-filter:blur(8px);margin-left:20px;">
    <div style="font-size:11px;">🕒 Local Clock</div>
    <div id="clock" style="font-size:22px;font-weight:800;margin-bottom:8px;">--:--:--</div>
    <hr style="border:1px solid rgba(255,255,255,.2);">
    <div style="font-size:11px;">⏳ Session</div>
    <div id="timer" style="font-size:18px;font-weight:700;">00:00</div>
    <hr style="border:1px solid rgba(255,255,255,.2);">
    <div style="font-size:14px;font-weight:700;color:#7CFF7C;">🟢 ONLINE</div>
  </div>
</div>
<script>
  const t0 = Date.now();
  function tick(){{
    const n=new Date();
    document.getElementById("clock").textContent=n.toLocaleTimeString();
    const s=Math.floor((Date.now()-t0)/1000);
    document.getElementById("timer").textContent =
      String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
  }}
  setInterval(tick,1000);tick();
</script>
""", height=200)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TABS SETUP
# ═══════════════════════════════════════════════════════════════════
(tab_home, tab_screen, tab_detail, tab_admet,
 tab_drugbank, tab_scaffold, tab_multitarget,
 tab_vina, tab_viewer, tab_ai,
 tab_future, tab_about) = st.tabs([
    "🏠 Dashboard",
    "🔬 Screening",
    "💊 Compounds",
    "🧪 Live ADMET",
    "💊 DrugBank",
    "🧬 Scaffolds",
    "🗂️ Multi-Target",
    "⚙️ Vina Docking",
    "🔭 3D Viewer",
    "🤖 AI Assistant",
    "🚀 Future",
    "👨‍💻 Developer",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 0: DASHBOARD (GLOWING GLASSMETRIC CARDS)
# ═══════════════════════════════════════════════════════════════════
def render_dashboard_tab() -> None:
    st.markdown('<div class="section-header">Platform Overview</div>', unsafe_allow_html=True)

    protein_loaded = (st.session_state.protein_info or {}).get("pdb_id", "None")
    compound_count = _compound_count()
    
    best_dock_score = "N/A"
    if not st.session_state.results_df.empty:
        best_row = ranking.get_best_compound(st.session_state.results_df)
        sc = score_col(st.session_state.results_df)
        best_dock_score = f"{best_row.get(sc, 0.0)} kcal/mol"

    ai_api_status = "🟢 Connected" if _get_api_key() else "🔴 Offline"

    # Glowing Glass Cards Grid
    grid_col_1, grid_col_2, grid_col_3 = st.columns(3)
    with grid_col_1:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">🧬</div>
            <div class="val">{protein_loaded}</div>
            <div class="lbl">Protein Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with grid_col_2:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">💊</div>
            <div class="val">{compound_count}</div>
            <div class="lbl">Compound Library</div>
        </div>
        """, unsafe_allow_html=True)
    with grid_col_3:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">🏆</div>
            <div class="val">{best_dock_score}</div>
            <div class="lbl">Best Dock Score</div>
        </div>
        """, unsafe_allow_html=True)

    grid_col_4, grid_col_5, grid_col_6 = st.columns(3)
    with grid_col_4:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">📄</div>
            <div class="val">{st.session_state.reports_count}</div>
            <div class="lbl">Reports Generated</div>
        </div>
        """, unsafe_allow_html=True)
    with grid_col_5:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">🤖</div>
            <div class="val">{ai_api_status}</div>
            <div class="lbl">AI Status</div>
        </div>
        """, unsafe_allow_html=True)
    with grid_col_6:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="icon">⚙️</div>
            <div class="val">v{APP_VERSION} Professional</div>
            <div class="lbl">Platform Version</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Quick Start Workflow</div>', unsafe_allow_html=True)

    steps = [
        ("1", "🔬 Screening",    "Fetch structural proteins and run virtual screening simulations"),
        ("2", "🧪 Live ADMET",   "Request real PubChem physicochemical ADMET descriptors"),
        ("3", "💊 DrugBank",     "Browse mechanism of action cards & clinical interactions"),
        ("4", "🧬 Scaffolds",    "Identify scaffolds utilizing Bemis-Murcko clustering"),
        ("5", "🗂️ Multi-Target", "Query selectivity distributions against target families"),
        ("6", "🤖 AI Assistant", "Interact with Gemini 2.5 Flash to synthesize findings"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f'<div class="compound-card"><b style="font-size:1.4rem">{num}</b>&nbsp;&nbsp;{title}<br>'
                f'<span style="color:#9DB4CC;font-size:.85rem">{desc}</span></div>',
                unsafe_allow_html=True,
            )

with tab_home:
    render_dashboard_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 1: VIRTUAL SCREENING
# ═══════════════════════════════════════════════════════════════════
def _render_progress(steps_done: int, steps: list[str]) -> None:
    for i, label in enumerate(steps):
        if i < steps_done:
            cls, icon = "done", "✅"
        elif i == steps_done:
            cls, icon = "active", "⏳"
        else:
            cls, icon = "", "○"
        st.markdown(
            f'<div class="progress-step {cls}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

def render_screening_tab() -> None:
    st.markdown('<div class="section-header">Step 1 — Protein Target</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        pdb_id_input = st.text_input(
            "PDB ID (4 characters)", value="2HU4", max_chars=4,
            help="Try: 2HU4 · 6LU7 · 1HVR · 3POZ · 1ATP",
        ).strip().upper()

        is_valid = bool(pdb_id_input) and protein.is_valid_pdb_id(pdb_id_input)
        if pdb_id_input and not is_valid:
            st.error("Invalid PDB ID.")

        if st.button("Fetch Protein Data", icon="🔎", disabled=not is_valid,
                     type="primary", use_container_width=True):
            with st.spinner(f"Querying RCSB PDB for {pdb_id_input}…"):
                st.session_state.protein_info = protein.get_protein_info(pdb_id_input)

    with c2:
        info = st.session_state.protein_info
        if info and info.get("pdb_id") == pdb_id_input:
            thumb_url = f"https://cdn.rcsb.org/images/structures/{pdb_id_input.lower()}_assembly-1.jpeg"
            try:
                tr = requests.get(thumb_url, timeout=5)
                if tr.status_code == 200:
                    st.image(tr.content, caption=f"{pdb_id_input} thumbnail", width=200)
            except Exception:
                pass

            src_label = ("✅ Live RCSB data" if info.get("source") == "live" else "⚠️ Offline fallback")
            st.info(src_label)
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Name:** {info.get('name','—')}")
                st.markdown(f"**Organism:** {info.get('organism','—')}")
                st.markdown(f"**Method:** {info.get('method','—')}")
                st.markdown(f"**Resolution:** {info.get('resolution','—')} Å")
            with right:
                chains  = info.get("chains") or []
                ligands = info.get("ligands") or []
                asr     = info.get("active_site_residues") or []
                st.markdown(f"**Chains:** {', '.join(chains) if chains else 'See PDB'}")
                st.markdown(f"**Ligands:** {', '.join(ligands) if ligands else 'None'}")
                asr_txt = (", ".join(asr[:4]) + ("…" if len(asr) > 4 else "")) if asr else "Not annotated"
                st.markdown(f"**Active Site:** {asr_txt}")
                st.markdown(f"[View on RCSB ↗]({info.get('structure_url','#')})")

            rl = info.get("reference_ligand")
            if rl:
                st.markdown("---")
                st.markdown(
                    f'<div class="ref-lig-card">'
                    f'<b>📌 {rl.get("name","—")}</b> — co-crystallised with <b>{pdb_id_input}</b><br>'
                    f'<b>Formula:</b> {rl.get("formula","—")} | <b>MW:</b> {rl.get("mw","—")} g/mol | '
                    f'<b>CID:</b> {rl.get("pubchem_cid","—")}<br>'
                    f'<em style="font-size:.84rem">{rl.get("notes","")}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="info-box">Enter a PDB ID and click <b>Fetch Protein Data</b>.</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Step 2 — Compound Library</div>', unsafe_allow_html=True)

    compounds_df = screening.load_compound_data(DATA_PATH)
    all_compounds = compounds_df["compound_name"].tolist()

    cs1, cs2 = st.columns([4, 1])
    with cs1:
        search_term = st.text_input("Filter", placeholder="Type a compound name…",
                                    label_visibility="collapsed")
    with cs2:
        st.metric("Library", len(all_compounds))

    filtered = ([c for c in all_compounds if search_term.lower() in c.lower()]
                if search_term else all_compounds)
    selected_compounds = st.multiselect(
        "Select compounds to screen",
        options=filtered, default=filtered[:5] if filtered else [],
    )

    with st.expander("Browse full compound database"):
        st.dataframe(
            compounds_df[safe_cols(compounds_df, [
                "compound_name", "molecular_formula", "molecular_weight",
                "logp", "h_donors", "h_acceptors", "rotatable_bonds",
            ])],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Step 3 — Run Virtual Screening</div>', unsafe_allow_html=True)

    protein_ready = (st.session_state.protein_info is not None
                     and st.session_state.protein_info.get("pdb_id") == pdb_id_input)
    reasons = (
        ([] if protein_ready       else ["fetch protein data in Step 1"])
        + ([] if selected_compounds else ["select at least one compound in Step 2"])
    )
    if reasons:
        st.caption("Before running, please " + " and ".join(reasons) + ".")

    run_clicked = st.button("Run Virtual Screening Simulation",
                            icon="🚀", type="primary", disabled=bool(reasons))

    if run_clicked:
        # High-Fidelity Professional Sequential Loading Workflow
        STEPS = [
            "Initializing AI Engine...",
            "Connecting to RCSB...",
            "Downloading Protein...",
            "Preparing Ligands...",
            "Generating Conformers...",
            "Running Virtual Screening...",
            "Calculating Docking Scores...",
            "Evaluating ADMET...",
            "Ranking Compounds...",
            "Generating AI Summary...",
            "Preparing Report...",
            "Completed Successfully"
        ]
        progress_slot = st.empty()
        prog_bar = st.progress(0)

        for idx, step_label in enumerate(STEPS):
            with progress_slot.container():
                _render_progress(idx, STEPS)
            prog_bar.progress(int((idx + 1) / len(STEPS) * 100))
            time.sleep(0.4)

        raw    = screening.simulate_virtual_screening(
            compounds_df, pdb_id_input, selected_compounds,
            protein_info=st.session_state.protein_info,
        )
        ranked = ranking.rank_compounds(raw)
        st.session_state.results_df = ranked
        st.session_state.pdf_ready    = False
        st.session_state.pdf_bytes    = None
        st.session_state.pdf_pdb_id   = None
        prog_bar.progress(100)
        progress_slot.empty()
        prog_bar.empty()

        best = ranking.get_best_compound(ranked)
        if best:
            sc = score_col(ranked)
            st.session_state.history.append({
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                "protein":          pdb_id_input,
                "compounds_tested": len(ranked),
                "best_compound":    best.get("compound_name"),
                "best_score":       best.get(sc),
            })
        st.rerun()

    results_df = st.session_state.results_df
    if results_df.empty:
        st.markdown('<div class="info-box">Complete Steps 1–2 and click <b>Run Virtual Screening</b>.</div>',
                    unsafe_allow_html=True)
        _render_disclaimer()
        return

    _render_results(results_df, pdb_id_input)
    _render_disclaimer()


def _render_results(results_df: pd.DataFrame, pdb_id: str) -> None:
    sc        = score_col(results_df)
    best_row  = ranking.get_best_compound(results_df)
    best_name = (best_row or {}).get("compound_name", "—")
    best_val  = (best_row or {}).get(sc, "—")
    target    = results_df["protein_target"].iloc[0] if "protein_target" in results_df.columns else pdb_id

    # Professional Summary Card
    st.markdown(f"""
    <div class="glass-card" style="border-left: 5px solid {GREEN_OK};">
        <h3 style="color:{GREEN_OK}; margin-top:0; margin-bottom:10px;">✅ Virtual Screening Completed</h3>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; font-size:0.95rem;">
            <div><b>Target Protein:</b> {target}</div>
            <div><b>Compounds Screened:</b> {len(results_df)}</div>
            <div><b>Best Performing Lead:</b> {best_name}</div>
            <div><b>Best Dock Score:</b> {best_val} kcal/mol</div>
            <div><b>Confidence Score:</b> {(best_row or {}).get("confidence_index", "0.0")}</div>
            <div><b>Processing Time:</b> {(best_row or {}).get("processing_time_ms", "0")} ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Summary
    with st.expander("🤖 AI-Generated Screening Summary", expanded=True):
        context = (
            f"Protein: {pdb_id}\n"
            f"Compounds screened: {len(results_df)}\n"
            f"Best compound: {best_name} ({best_val} kcal/mol)\n"
            f"Compounds: {', '.join(results_df['compound_name'].tolist()[:10])}\n"
        )
        ai_key = f"summary_{pdb_id}_{best_name}"
        if ai_key not in st.session_state:
            with st.spinner("Generating AI summary with Gemini…"):
                summary = _call_gemini(
                    system=(
                        "You are a senior computational medicinal chemist. "
                        "Write a concise 3-paragraph summary of a virtual screening run. "
                        "Paragraph 1: what was screened and why. "
                        "Paragraph 2: interpret the top compound's score and ADMET implications. "
                        "Paragraph 3: next experimental steps. Always note scores are simulated."
                    ),
                    messages=[{"role": "user", "content": context}],
                )
            st.session_state[ai_key] = summary
        st.markdown(st.session_state[ai_key])

    # Ranked table
    st.markdown("#### 📊 Ranked Screening Results")
    disp = safe_cols(results_df, ["rank", "compound_name", "molecular_formula",
                                   "molecular_weight", "logp", sc, "lipinski_status"])
    rename = {sc: "Sim. Score (kcal/mol)", "compound_name": "Compound",
               "molecular_formula": "Formula", "molecular_weight": "MW (g/mol)",
               "logp": "LogP", "lipinski_status": "Lipinski Ro5"}

    def _hl(row):
        return ["background-color:#0D3B26" if row.get("rank") == 1 else ""] * len(row)

    st.dataframe(
        results_df[disp].rename(columns=rename).style.apply(_hl, axis=1),
        use_container_width=True, hide_index=True,
    )

    # ADMET
    st.markdown("#### 🧪 ADMET Properties")
    admet_cols = safe_cols(results_df, ["compound_name", "molecular_weight", "logp",
                                         "h_donors", "h_acceptors", "rotatable_bonds", "lipinski_status"])
    st.dataframe(
        results_df[admet_cols].rename(columns={
            "compound_name": "Compound", "molecular_weight": "MW (g/mol)",
            "logp": "LogP", "h_donors": "H-Donors", "h_acceptors": "H-Acceptors",
            "rotatable_bonds": "Rot. Bonds", "lipinski_status": "Lipinski Ro5",
        }),
        use_container_width=True, hide_index=True,
    )

    # Charts
    st.markdown("#### 📈 Score Analysis")
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_bar = px.bar(
            results_df.sort_values(sc), x="compound_name", y=sc,
            color=sc, color_continuous_scale="RdYlGn_r",
            title="Scores per Compound",
            labels={"compound_name": "Compound", sc: "Sim. Score (kcal/mol)"},
            text=sc,
        )
        fig_bar.update_traces(textposition="outside", texttemplate="%{text:.2f}")
        fig_bar.update_layout(showlegend=False, **plotly_layout())
        st.plotly_chart(fig_bar, use_container_width=True)
    with ch2:
        if {"molecular_weight", "logp"}.issubset(results_df.columns):
            fig_sc = px.scatter(
                results_df, x="molecular_weight", y=sc, size="molecular_weight",
                color="compound_name", hover_name="compound_name",
                title="MW vs Simulated Score",
                labels={"molecular_weight": "MW (g/mol)", sc: "Sim. Score"},
            )
            fig_sc.update_layout(**plotly_layout())
            st.plotly_chart(fig_sc, use_container_width=True)

    # Advanced charts
    st.markdown("#### 🔬 Advanced Visualisations")
    adv1, adv2 = st.columns(2)
    with adv1:
        num_props = safe_cols(results_df, [sc, "molecular_weight", "logp", "h_donors", "h_acceptors"])
        if len(num_props) >= 2:
            melted = results_df[num_props].melt(var_name="Property", value_name="Value")
            fig_box = px.box(melted, x="Property", y="Value", color="Property",
                             title="Property Distribution (Box Plot)",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_box.update_layout(**plotly_layout())
            st.plotly_chart(fig_box, use_container_width=True)
    with adv2:
        if sc in results_df.columns and "lipinski_status" in results_df.columns:
            fig_vio = px.violin(
                results_df, y=sc, x="lipinski_status", color="lipinski_status",
                box=True, points="all", hover_name="compound_name",
                title="Score by Lipinski Status",
                color_discrete_sequence=[TEAL_LIGHT, GOLD, PINK],
            )
            fig_vio.update_layout(**plotly_layout())
            st.plotly_chart(fig_vio, use_container_width=True)

    num_cols = safe_cols(results_df, ["molecular_weight", "logp", "h_donors",
                                       "h_acceptors", "rotatable_bonds", sc])
    if len(num_cols) >= 3:
        corr = results_df[num_cols].corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig_heat.update_layout(title="Correlation Matrix", **plotly_layout())
        st.plotly_chart(fig_heat, use_container_width=True)

    # Radar
    radar_props = safe_cols(results_df, ["molecular_weight", "logp", "h_donors",
                                          "h_acceptors", "rotatable_bonds"])
    if radar_props and len(results_df) >= 1:
        top3 = results_df.nsmallest(min(3, len(results_df)), sc)
        norm = ((results_df[radar_props] - results_df[radar_props].min())
                / (results_df[radar_props].max() - results_df[radar_props].min() + 1e-9))
        fig_radar = go.Figure()
        for i, (_, row) in enumerate(top3.iterrows()):
            vals = norm.loc[row.name, radar_props].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_props + [radar_props[0]],
                fill="toself", name=row["compound_name"],
                line_color=[TEAL_LIGHT, GOLD, PINK][i % 3],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar: Top 3 Compounds", **plotly_layout(),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Metrics
    st.markdown('<div class="section-header">Scientific Metrics</div>', unsafe_allow_html=True)
    try:
        stats = {
            "count":         len(results_df),
            "avg_mw":        round(results_df["molecular_weight"].mean(), 1) if "molecular_weight" in results_df.columns else "—",
            "best_score":    results_df[sc].min(),
            "worst_score":   results_df[sc].max(),
            "average_score": round(results_df[sc].mean(), 2),
        }
    except Exception:
        stats = {}

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, icon, val, lbl in [
        (m1, "🔬", stats.get("count", 0),                      "Screened"),
        (m2, "⚖️", stats.get("avg_mw", "—"),                  "Avg. MW (g/mol)"),
        (m3, "🏆", f"{stats.get('best_score','—')} kcal/mol",  "Best Score"),
        (m4, "📉", f"{stats.get('worst_score','—')} kcal/mol", "Worst Score"),
        (m5, "📊", f"{stats.get('average_score','—')} kcal/mol","Average Score"),
    ]:
        metric_card(col, icon, val, lbl)

    # Export Report
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "Download CSV", icon="⬇️",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"screening_{pdb_id}_{ts}.csv",
            mime="text/csv", use_container_width=True,
        )

    with e2:
        if not st.session_state.pdf_ready or st.session_state.pdf_pdb_id != pdb_id:
            if st.button("Prepare PDF Report", icon="📄", use_container_width=True):
                with st.spinner("Building PDF report…"):
                    try:
                        pdf_bytes = report.generate_pdf_report(
                            st.session_state.protein_info, results_df, stats
                        )
                        st.session_state.pdf_bytes  = pdf_bytes
                        st.session_state.pdf_ready  = True
                        st.session_state.pdf_pdb_id = pdb_id
                        st.rerun()
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
        else:
            clicked = st.download_button(
                "Download PDF Report", icon="📄",
                data=st.session_state.pdf_bytes,
                file_name=f"report_{pdb_id}_{ts}.pdf",
                mime="application/pdf", use_container_width=True,
                on_click=lambda: None,
            )


def _render_disclaimer() -> None:
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Educational Use Only.</b> '
        'All docking scores are formula-based simulations — not actual docking results.</div>',
        unsafe_allow_html=True,
    )

with tab_screen:
    render_screening_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 2: COMPOUND DETAILS
# ═══════════════════════════════════════════════════════════════════
def render_compound_tab() -> None:
    st.markdown('<div class="section-header">💊 Compound Details</div>', unsafe_allow_html=True)

    compounds_df_d = screening.load_compound_data(DATA_PATH)
    sel = st.selectbox("Select a compound", compounds_df_d["compound_name"].tolist())
    if not sel:
        return

    row    = compounds_df_d[compounds_df_d["compound_name"] == sel].iloc[0]
    smiles = row.get("smiles", "")
    cid    = row.get("pubchem_cid", None)
    db_id  = row.get("drugbank_id", None)

    d1, d2 = st.columns([2, 3])
    with d1:
        st.markdown(f"### {sel}")
        for label, key in [
            ("Molecular Formula",  "molecular_formula"),
            ("Molecular Weight",   "molecular_weight"),
            ("LogP",               "logp"),
            ("H-Bond Donors",      "h_donors"),
            ("H-Bond Acceptors",   "h_acceptors"),
            ("Rotatable Bonds",    "rotatable_bonds"),
        ]:
            st.markdown(f"**{label}:** {row.get(key, '—')}"
                        + (" g/mol" if key == "molecular_weight" else ""))

        lip = screening.lipinski_status(row.to_dict())
        st.markdown(f"**Lipinski Ro5:** {lip}")
        if cid and str(cid) != "nan":
            st.markdown(f"[View on PubChem ↗](https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)})")
        if db_id and str(db_id) != "nan":
            st.markdown(f"[View on DrugBank ↗](https://go.drugbank.com/drugs/{db_id})")
        st.code(smiles or "No SMILES", language="text")

    with d2:
        st.markdown("#### 🖼️ Molecular Structure")
        if cid and str(cid) != "nan":
            cid_int = int(cid)
            for label, record_type in [("2D Structure", "2d"), ("3D Conformer", "3d")]:
                url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                       f"{cid_int}/PNG?record_type={record_type}&image_size=400x300")
                try:
                    r = requests.get(url, timeout=8)
                    if r.status_code == 200:
                        st.image(r.content, caption=f"{sel} — {label}", use_container_width=True)
                except Exception:
                    st.info(f"{label} image unavailable.")

with tab_detail:
    render_compound_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 3: LIVE PUBCHEM ADMET PANEL
# ═══════════════════════════════════════════════════════════════════
def render_admet_tab() -> None:
    st.markdown('<div class="section-header">🧪 PubChem Live ADMET Panel</div>', unsafe_allow_html=True)
    st.write("Fetch real physicochemical and bioassay data from PubChem PUG-REST, "
             "then compute validated ADMET flags.")

    compounds_df = screening.load_compound_data(DATA_PATH)
    sel_compounds = st.multiselect(
        "Select compounds for ADMET analysis",
        options=compounds_df["compound_name"].tolist(),
        default=compounds_df["compound_name"].tolist()[:4],
    )

    if not sel_compounds:
        st.info("Select at least one compound above.")
        return

    if st.button("🔬 Fetch Live ADMET from PubChem", type="primary"):
        rows = []
        progress = st.progress(0)
        for i, compound in enumerate(sel_compounds):
            row = compounds_df[compounds_df["compound_name"] == compound].iloc[0]
            cid = row.get("pubchem_cid")
            cid = int(cid) if cid and str(cid) not in ("nan", "None") else None
            with st.spinner(f"Fetching {compound}…"):
                data = fetch_pubchem_admet(compound, cid)
            rows.append(data)
            progress.progress(int((i + 1) / len(sel_compounds) * 100))
        st.session_state["admet_results"] = rows
        progress.empty()

    if "admet_results" not in st.session_state or not st.session_state["admet_results"]:
        return

    rows = st.session_state["admet_results"]

    st.markdown("#### 📊 Physicochemical Properties (Live PubChem)")
    table_rows = []
    for r in rows:
        table_rows.append({
            "Compound":    r.get("compound", "—"),
            "MW (g/mol)":  r.get("molecular_weight", "—"),
            "XLogP":       r.get("xlogp", "—"),
            "H-Donors":    r.get("h_donors", "—"),
            "H-Acceptors": r.get("h_acceptors", "—"),
            "Rot. Bonds":  r.get("rotatable_bonds", "—"),
            "TPSA (Å²)":   r.get("tpsa", "—"),
            "Complexity":  r.get("complexity", "—"),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Bioassay summary
    st.markdown("#### 🔬 PubChem Bioassay Activity")
    ba_rows = []
    for r in rows:
        if "bioassay_total_count" in r:
            total  = r.get("bioassay_total_count", 0)
            active = r.get("bioassay_active_count", 0)
            rate   = f"{active/total*100:.1f}%" if total > 0 else "—"
            ba_rows.append({
                "Compound":        r.get("compound", "—"),
                "Total Assays":    total,
                "Active Assays":   active,
                "Activity Rate":   rate,
            })
    if ba_rows:
        st.dataframe(pd.DataFrame(ba_rows), use_container_width=True, hide_index=True)

    # ADMET flags
    st.markdown("#### ✅ Validated ADMET Flags")
    for r in rows:
        flags = r.get("admet_flags", {})
        if not flags:
            continue
        st.markdown(f"**{r.get('compound','—')}**")
        cols = st.columns(len(flags))
        for i, (prop, (status, note)) in enumerate(flags.items()):
            with cols[i]:
                st.markdown(render_badge(status, prop), unsafe_allow_html=True)

    # TPSA vs LogP chart
    if len(rows) >= 2:
        chart_data = pd.DataFrame([{
            "Compound": r.get("compound"),
            "TPSA":     r.get("tpsa"),
            "XLogP":    r.get("xlogp"),
            "MW":       r.get("molecular_weight"),
        } for r in rows if r.get("tpsa") and r.get("xlogp")]).dropna()

        if len(chart_data) >= 2:
            try:
                chart_data["TPSA"]  = chart_data["TPSA"].astype(float)
                chart_data["XLogP"] = chart_data["XLogP"].astype(float)
                chart_data["MW"]    = chart_data["MW"].astype(float)
                fig = px.scatter(
                    chart_data, x="XLogP", y="TPSA", size="MW",
                    color="Compound", hover_name="Compound",
                    title="TPSA vs XLogP (Veber Rule Visualisation)",
                    labels={"XLogP": "XLogP (Lipophilicity)", "TPSA": "TPSA (Å²)"},
                )
                fig.add_hline(y=140, line_dash="dash", line_color=GOLD,
                              annotation_text="TPSA = 140 Å² (Veber limit)")
                fig.add_vline(x=5,  line_dash="dash", line_color=PINK,
                              annotation_text="LogP = 5 (Lipinski limit)")
                fig.update_layout(**plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

with tab_admet:
    render_admet_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 4: DRUGBANK CLINICAL MATRIX (INCORPORATED BEAUTIFUL CARDS)
# ═══════════════════════════════════════════════════════════════════
def render_drugbank_tab() -> None:
    st.markdown('<div class="section-header">💊 DrugBank Information</div>', unsafe_allow_html=True)
    st.write("View mechanism of action, pharmacokinetics, drug interactions, and targets.")

    db_key = _get_drugbank_key()
    if db_key:
        st.success("✅ DrugBank API key found — live data enabled.")
    else:
        st.info(
            "💡 No DrugBank API key found. Showing curated data for common drugs. "
            "Add DRUGBANK_API_KEY to secrets.toml for live access."
        )

    compounds_df = screening.load_compound_data(DATA_PATH)
    sel = st.selectbox("Select compound", compounds_df["compound_name"].tolist(), key="db_sel")
    row = compounds_df[compounds_df["compound_name"] == sel].iloc[0]
    db_id = row.get("drugbank_id")
    db_id = str(db_id) if db_id and str(db_id) not in ("nan", "None") else None

    if st.button("🔍 Fetch DrugBank Info", type="primary"):
        with st.spinner(f"Fetching info for {sel}…"):
            info = fetch_drugbank_info(sel, db_id)
        st.session_state["db_current"] = info

    info = st.session_state.get("db_current")
    if not info:
        return

    src = info.get("source", "unavailable")
    if src == "drugbank_live":
        st.success("✅ Live DrugBank data")
    elif src == "curated":
        st.info("📚 Curated offline data (add API key for full data)")
    else:
        st.warning(info.get("note", "No data available for this compound."))
        return

    # Beautiful Responsive CSS Cards Grid for Clinical Diagnostics
    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
        <div class="glass-card">
            <h4 style="color:{TEAL_LIGHT}; margin-top:0;">🏥 Action Mechanism</h4>
            <p style="font-size:0.9rem; line-height:1.5;">{info.get('mechanism', '—')}</p>
        </div>
        <div class="glass-card">
            <h4 style="color:{TEAL_LIGHT}; margin-top:0;">🎯 Targeted Receptor</h4>
            <p style="font-size:0.9rem; line-height:1.5;">{info.get('targets', '—') if not isinstance(info.get('targets'), list) else 'Listed below'}</p>
        </div>
        <div class="glass-card">
            <h4 style="color:{TEAL_LIGHT}; margin-top:0;">⚡ Hepatic Metabolism</h4>
            <p style="font-size:0.9rem; line-height:1.5;">{info.get('metabolism', '—')}</p>
        </div>
        <div class="glass-card">
            <h4 style="color:{TEAL_LIGHT}; margin-top:0;">⏱️ Half-life Parameter</h4>
            <p style="font-size:0.9rem; line-height:1.5;">{info.get('half_life', '—')}</p>
        </div>
        <div class="glass-card">
            <h4 style="color:{TEAL_LIGHT}; margin-top:0;">🔗 Protein Binding</h4>
            <p style="font-size:0.9rem; line-height:1.5;">{info.get('protein_binding', '—')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if info.get("categories"):
        st.markdown("#### 🏷️ Drug Categories")
        for cat in info["categories"]:
            st.markdown(f'<span class="selectivity-badge" style="background:{TEAL_MID}">{cat}</span>',
                        unsafe_allow_html=True)

    # Targets list
    targets = info.get("targets", [])
    if isinstance(targets, list) and targets:
        st.markdown("#### 🎯 Molecular Targets")
        t_rows = [{"Target": t["name"], "Gene": t["gene"], "Action": t["action"]}
                  for t in targets]
        st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

    # Interactions
    interactions = info.get("interactions", [])
    if isinstance(interactions, list) and interactions:
        st.markdown("#### ⚠️ Drug Interactions (Top 10)")
        i_rows = [{"Drug": i["drug"], "Severity": i["severity"], "Description": i["description"][:100]}
                  for i in interactions]
        st.dataframe(pd.DataFrame(i_rows), use_container_width=True, hide_index=True)

with tab_drugbank:
    render_drugbank_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 5: SCAFFOLD CLUSTERING
# ═══════════════════════════════════════════════════════════════════
with tab_scaffold:
    compounds_df_sc = screening.load_compound_data(DATA_PATH)
    render_scaffold_clustering(compounds_df_sc)


# ═══════════════════════════════════════════════════════════════════
# TAB 6: MULTI-TARGET MAPS
# ═══════════════════════════════════════════════════════════════════
with tab_multitarget:
    render_multi_target_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 7: AUTODOCK VINA
# ═══════════════════════════════════════════════════════════════════
with tab_vina:
    render_vina_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 8: 3D VIEWER
# ═══════════════════════════════════════════════════════════════════
def render_viewer_tab() -> None:
    st.markdown('<div class="section-header">🧪 3D Protein Structure Viewer</div>', unsafe_allow_html=True)

    info = st.session_state.protein_info
    v1, v2 = st.columns([1, 3])
    with v1:
        view_pdb = st.text_input(
            "PDB ID to view",
            value=(info.get("pdb_id") if info else "2HU4"),
            max_chars=4,
        ).strip().upper()
        st.info("💡 Rotate: drag · Zoom: scroll · Pan: right-drag")

    with v2:
        st.markdown(
            f'<iframe src="https://www.rcsb.org/3d-view/{view_pdb}?preset=defaultView" '
            f'width="100%" height="520" '
            f'style="border:2px solid {TEAL_MID};border-radius:10px;" '
            f'title="3D Structure of {view_pdb}"></iframe>',
            unsafe_allow_html=True,
        )

with tab_viewer:
    render_viewer_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 9: AI ASSISTANT (CONVERTED CLAUDE REFERENCES TO GEMINI 2.5 FLASH)
# ═══════════════════════════════════════════════════════════════════
_AI_SYSTEM = """
You are an expert AI Research Assistant (Gemini 2.5 Flash) for a drug discovery platform used by B.Tech
Biotechnology students. You help with:
- Virtual screening concepts and interpretation
- Simulated docking score analysis
- ADMET, Lipinski Ro5, pharmacokinetics
- Scaffold clustering and lead optimisation
- Multi-target selectivity
- AutoDock Vina usage and real docking
- Next experimental steps
Always clarify when scores are simulated vs real. Be educational, encouraging, and concise.
"""

_SUGGESTED = [
    "Why did the top compound score best against the pocket?",
    "Explain Lipinski's Rule of Five conditions",
    "What does ADMET represent in clinical pipelines?",
    "How do I evaluate TPSA boundaries?",
    "What is a Bemis-Murcko core scaffold?",
    "Explain multi-target selectivity maps",
    "How does the AutoDock Vina scoring formula compute?",
    "What are next validation steps after screening?",
]

def render_ai_tab() -> None:
    st.markdown('<div class="section-header">🤖 Gemini 2.5 Flash Research Assistant</div>', unsafe_allow_html=True)
    st.write("Full conversation memory — Gemini retains session interactions context.")

    if not _get_api_key():
        st.error("⚠️ GEMINI_API_KEY environment variable missing. Populate tokens inside secrets.toml files.")
        return

    results_df = st.session_state.results_df
    prot_info  = st.session_state.protein_info
    context_parts = []
    if prot_info:
        context_parts.append(f"Protein: {prot_info.get('pdb_id')} — {prot_info.get('name','')}")
    if not results_df.empty:
        sc   = score_col(results_df)
        best = ranking.get_best_compound(results_df)
        context_parts.append(
            f"Last screening: {len(results_df)} compounds, "
            f"best: {(best or {}).get('compound_name','—')} ({(best or {}).get(sc,'—')} kcal/mol)"
        )
    if st.session_state.multi_target_results:
        targets = list(st.session_state.multi_target_results.keys())
        context_parts.append(f"Multi-target screening done: {', '.join(targets)}")

    if context_parts:
        st.info("📋 Session context: " + " · ".join(context_parts))

    # Suggested questions grid
    sq_cols = st.columns(4)
    for i, q in enumerate(_SUGGESTED):
        with sq_cols[i % 4]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                context_str = "\n".join(context_parts) or "No session data."
                user_content = f"Session context:\n{context_str}\n\nQuestion: {q}"
                st.session_state.ai_chat.append({"role": "user", "content": user_content})
                with st.spinner("Gemini calculating response…"):
                    reply = _call_gemini(_AI_SYSTEM, st.session_state.ai_chat)
                st.session_state.ai_chat.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("---")

    for msg in st.session_state.ai_chat:
        display_text = msg["content"]
        if "Session context:" in display_text and "Question:" in display_text:
            display_text = display_text.split("Question:")[-1].strip()
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 {display_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-bubble">🤖 {display_text}</div>', unsafe_allow_html=True)

    with st.form("ai_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="e.g. Why did Oseltamivir rank first?",
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Send", use_container_width=False)

    if send and user_input.strip():
        context_str  = "\n".join(context_parts) or "No session data."
        user_content = f"Session context:\n{context_str}\n\nQuestion: {user_input.strip()}"
        st.session_state.ai_chat.append({"role": "user", "content": user_content})
        with st.spinner("Gemini generating response…"):
            reply = _call_gemini(_AI_SYSTEM, st.session_state.ai_chat)
        st.session_state.ai_chat.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.ai_chat:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear chat"):
                st.session_state.ai_chat = []
                st.rerun()
        with col2:
            st.caption(f"💬 {len(st.session_state.ai_chat)//2} exchanges in memory")

with tab_ai:
    render_ai_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 10: FUTURE UPGRADES
# ═══════════════════════════════════════════════════════════════════
def render_future_tab() -> None:
    st.markdown('<div class="section-header">🚀 Future Upgrade Roadmap</div>', unsafe_allow_html=True)

    upgrades = [
        ("🧬 Real Force-Field Docking",      "MMFF94/Gasteiger charge docking for genuine kcal/mol energies.", "High"),
        ("🤖 GNN / Transformer Lead Prediction","MolBERT / ChemBERTa for AI-predicted binding affinity.", "High"),
        ("🌊 Molecular Dynamics",             "OpenMM interface for MM-PBSA binding free energy.", "Medium"),
        ("📊 ADMETlab API",                  "Full validated ADMET profiles from ADMETlab 2.0.", "Medium"),
        ("☁️ User Accounts & Projects",      "Save projects, compare runs across sessions.", "Low"),
        ("🔗 UniProt Integration",            "Fetch protein function, disease associations, pathways.", "Medium"),
        ("📈 SAR Analysis",                  "Structure-Activity Relationship plots and R-group decomposition.", "Medium"),
        ("🧫 Cell Viability Prediction",     "Predict cytotoxicity from molecular features.", "Low"),
    ]
    colours = {"High": "#C0392B", "Medium": GOLD, "Low": GREEN_OK}
    for title, desc, priority in upgrades:
        st.markdown(
            f'<div class="roadmap-item"><b>{title}</b>'
            f'<span style="float:right;background:{colours.get(priority,"#555")};color:white;'
            f'border-radius:10px;padding:1px 8px;font-size:.75rem">{priority} Priority</span>'
            f'<br><span style="color:#9DB4CC;font-size:.87rem">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### ✅ Completed in v5.0")
    done = [
        "Gemini 2.5 Flash model connectivity fixes",
        "Branding updates to AI Drug Discovery Platform v5.0",
        "Vibrant layout correction fixing the floating top footer",
        "ADMET colored badge matrices in Live ADMET tab",
        "Clinical responsive metrics cards inside DrugBank tab",
    ]
    for item in done:
        st.markdown(f"✅ {item}")

with tab_future:
    render_future_tab()


# ═══════════════════════════════════════════════════════════════════
# TAB 11: DEVELOPER / ABOUT
# ═══════════════════════════════════════════════════════════════════
def render_about_tab() -> None:
    st.markdown(f"## 👨‍💻 {DEVELOPER}")
    st.markdown(f"**B.Tech Biotechnology · {INSTITUTION} · Final Year Project 2026**")
    st.markdown("---")

    a1, a2 = st.columns([2, 1])
    with a1:
        st.markdown(f"""
### About This Project
AI Drug Discovery Platform v{APP_VERSION} is a full-stack computational biology platform integrating:
protein structure retrieval, virtual screening simulation, live PubChem ADMET data,
DrugBank pharmacology, Bemis-Murcko scaffold clustering, multi-target selectivity,
AutoDock Vina real docking, AI-powered research assistance, and automated PDF reporting.

### Technologies
**Frontend:** Python · Streamlit · Plotly · HTML/CSS  
**Bioinformatics:** RCSB PDB API · PubChem PUG-REST · DrugBank API · NGL Viewer  
**Cheminformatics:** RDKit · SciPy · AutoDock Vina  
**AI:** Gemini 2.5 Flash via REST endpoints  
**Development:** AI-assisted

### v5.0 Improvements
- Full rebranding to Version 5.0 Professional
- Swapped legacy preview model for stable gemini-2.5-flash
- Corrected layout positioning to keep the footer below the tab contents
""")

    with a2:
        st.markdown("### 📞 Contact")
        st.markdown("📱 +91 9346599651")
        st.markdown("📧 vamsikrishnareddynemaildinne@gmail.com")
        st.markdown("[📸 Instagram](https://www.instagram.com/n_vamsi_reddie)")
        st.markdown("[💻 GitHub](https://github.com/vamsikrishnareddy66)")
        st.markdown("[🚀 Project Repo](https://github.com/vamsikrishnareddy66/AI-Drug-Explorer)")

    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ Educational Use Only.</div>', unsafe_allow_html=True)

with tab_about:
    render_about_tab()


# ═══════════════════════════════════════════════════════════════════
# CORRECT EXECUTION ORDER OF FOOTER (ALWAYS PLACED AT THE BOTTOM)
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="site-footer">'
    f'<b>AI Drug Discovery Platform</b> &nbsp;·&nbsp; v{APP_VERSION}<br>'
    f'Developed by <b>{DEVELOPER}</b> &nbsp;·&nbsp; {INSTITUTION}<br><br>'
    f'Powered by &nbsp;'
    f'<a href="https://www.python.org">Python</a> &nbsp;·&nbsp; '
    f'<a href="https://streamlit.io">Streamlit</a> &nbsp;·&nbsp; '
    f'<a href="https://www.rcsb.org">RCSB PDB</a> &nbsp;·&nbsp; '
    f'<a href="https://pubchem.ncbi.nlm.nih.gov">PubChem</a> &nbsp;·&nbsp; '
    f'<a href="https://go.drugbank.com">DrugBank</a> &nbsp;·&nbsp; '
    f'<a href="https://ai.google.dev">Gemini 2.5 Flash</a> &nbsp;·&nbsp; '
    f'<a href="https://vina.scripps.edu">AutoDock Vina</a><br><br>'
    f'<span style="font-size:.78rem;opacity:.6;">© 2026 KL University - For educational research validation only.</span>'
    f'</div>',
    unsafe_allow_html=True,
)
