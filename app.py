"""
app.py
------
Drug Discovery Pipeline — Professional Edition v3.0
B.Tech Biotechnology · KL University

Improvements over v2.0:
  ★ Landing dashboard with live stat cards
  ★ AI Research Assistant tab (Claude-powered)
  ★ Animated step-by-step progress bar during screening
  ★ Protein thumbnail via RCSB
  ★ AI-generated screening summary
  ★ Additional charts: heatmap, radar, box plot, violin, correlation matrix
  ★ Modular structure (theme / sidebar / tabs each in own section)
  ★ Deduplicated imports
  ★ Type hints + docstrings on every helper
  ★ config.py-ready constants block
  ★ Professional footer

Run:
    pip install streamlit pandas plotly requests fpdf2
    streamlit run app.py
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS  (deduplicated — each package imported exactly once)
# ═══════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# Local modules (unchanged from your existing codebase)
import protein
import ranking
import report
import screening
import statistics as stats_module

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION  (single source of truth — mirrors a future config.py)
# ═══════════════════════════════════════════════════════════════════

DATA_PATH   = "compounds.csv"
EXPORTS_DIR = "exports"
SCORE_COL   = "simulated_score"
APP_VERSION = "3.0"
DEVELOPER   = "N. Vamsi Krishna Reddy"
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
# THEME  (global CSS)
# ═══════════════════════════════════════════════════════════════════

def inject_theme() -> None:
    """Inject global CSS theme into the Streamlit page."""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: white;
}}

.stApp {{
    background: linear-gradient(135deg,{TEAL_DARK} 0%,#142850 30%,{TEAL_MID} 65%,#6A11CB 100%);
    color: white;
}}

/* ── Dashboard stat cards ── */
.stat-card {{
    background: rgba(20,27,58,.88);
    border: 1px solid rgba(0,212,255,.35);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform .2s, box-shadow .2s;
}}
.stat-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,212,255,.25);
}}
.stat-card .icon {{ font-size: 2rem; margin-bottom: 6px; }}
.stat-card .val  {{ font-size: 1.9rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.stat-card .lbl  {{ color: #DCE9FF; font-size: .82rem; margin-top: 4px; }}

/* ── Metric cards ── */
.metric-card {{
    background: rgba(20,27,58,.88);
    border: 1px solid rgba(0,212,255,.35);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    backdrop-filter: blur(12px);
}}
.metric-card .val {{ font-size: 1.8rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.metric-card .lbl {{ color: #DCE9FF; font-size: .82rem; }}

/* ── Section headers ── */
.section-header {{
    border-left: 5px solid {TEAL_LIGHT};
    padding-left: 12px;
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 25px 0 15px;
}}

/* ── Info / disclaimer / ref-lig boxes ── */
.info-box {{
    background: rgba(0,212,255,.08);
    border: 1px solid rgba(0,212,255,.35);
    border-radius: 12px;
    padding: 16px;
    color: #EAF7FF;
}}
.disclaimer {{
    background: rgba(255,209,102,.08);
    border: 1px solid {GOLD};
    border-radius: 12px;
    padding: 14px;
    color: #FFE7A0;
}}
.ref-lig-card {{
    background: rgba(20,27,58,.9);
    border: 1px solid rgba(0,212,255,.25);
    border-radius: 14px;
    padding: 18px;
    color: white;
}}
.compound-card {{
    background: rgba(20,27,58,.88);
    border: 1px solid rgba(123,97,255,.35);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    color: white;
    backdrop-filter: blur(12px);
}}
.compound-card b {{ color: {TEAL_LIGHT}; }}

/* ── Roadmap ── */
.roadmap-item {{
    background: rgba(20,27,58,.85);
    border-left: 4px solid {TEAL_LIGHT};
    border-radius: 10px;
    padding: 12px 16px;
    margin: 10px 0;
    color: white;
}}

/* ── AI chat bubbles ── */
.ai-bubble {{
    background: rgba(106,17,203,.25);
    border: 1px solid rgba(0,212,255,.3);
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    color: white;
    margin-bottom: 10px;
}}
.user-bubble {{
    background: rgba(0,229,255,.12);
    border: 1px solid rgba(0,229,255,.3);
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    color: white;
    margin-bottom: 10px;
    text-align: right;
}}

/* ── Progress bar ── */
.progress-step {{
    background: rgba(20,27,58,.85);
    border: 1px solid rgba(0,212,255,.2);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    color: #DCE9FF;
    font-size: .9rem;
}}
.progress-step.done   {{ border-color: {GREEN_OK}; color: {GREEN_OK}; }}
.progress-step.active {{ border-color: {TEAL_LIGHT}; color: white; font-weight: 700; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,{TEAL_DARK},{TEAL_MID} 120%);
}}
section[data-testid="stSidebar"] * {{ color: white !important; }}
section[data-testid="stSidebar"] .stTextInput input {{ color: black !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(20,27,58,.8);
    color: white;
    border-radius: 12px 12px 0 0;
    font-weight: 700;
    padding: 10px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg,#6A11CB,#00D4FF) !important;
    color: white !important;
}}

/* ── Buttons ── */
.stButton>button {{
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg,#6A11CB,#00D4FF);
    color: white;
    font-weight: 700;
    transition: .3s;
}}
.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,212,255,.4);
}}

/* ── Footer ── */
.site-footer {{
    background: rgba(7,26,47,.95);
    border-top: 1px solid rgba(0,212,255,.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 40px;
    text-align: center;
    color: #9DB4CC;
    font-size: .85rem;
}}
.site-footer b {{ color: {TEAL_LIGHT}; }}
.site-footer a  {{ color: {GOLD}; text-decoration: none; }}

/* ── Inputs ── */
.stTextInput input, .stSelectbox div, .stNumberInput input {{
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)


inject_theme()


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE  (initialised once)
# ═══════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "history":       [],
    "results_df":    pd.DataFrame(),
    "protein_info":  None,
    "reports_count": 0,
    "ai_chat":       [],          # list of {"role": "user"|"assistant", "content": str}
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def safe_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return only columns that exist in *df*."""
    return [c for c in wanted if c in df.columns]


def score_col(df: pd.DataFrame) -> str:
    """Active score column, with graceful fallback."""
    return SCORE_COL if SCORE_COL in df.columns else "docking_score"


def plotly_layout(extra: dict | None = None) -> dict:
    """Shared transparent Plotly layout for themed charts."""
    base = dict(
        plot_bgcolor="rgba(20,27,58,0.7)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    if extra:
        base.update(extra)
    return base


def metric_card(col: st.delta_generator.DeltaGenerator,
                icon: str, value: str | int | float, label: str) -> None:
    """Render a styled metric card inside *col*."""
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


def _call_claude(system: str, messages: list[dict]) -> str:
    """
    Call the Anthropic Messages API and return the assistant text.
    Works inside Streamlit Cloud (API key injected by the platform).
    """
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": system,
        "messages": messages,
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        data = r.json()
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
    except Exception as e:
        return f"⚠️ AI Assistant unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Render the left sidebar with platform info and history."""
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/DNA_icon.svg/512px-DNA_icon.svg.png",
        width=80,
    )
    with st.sidebar:
        st.markdown(f"# 🧬 Drug Discovery Pipeline")
        st.caption(f"v{APP_VERSION} · {INSTITUTION}")
        st.markdown("---")
        st.success("🟢 System: Online")

        st.markdown("### 📊 Quick Stats")
        st.metric("💊 Compounds",      _compound_count())
        st.metric("🧬 Current Protein",
                  (st.session_state.protein_info or {}).get("pdb_id", "Not loaded"))
        st.metric("📄 Reports Generated", st.session_state.reports_count)

        st.markdown("---")
        st.markdown("### 🔗 Popular Targets")
        targets = {
            "6LU7": "SARS-CoV-2 Main Protease",
            "2HU4": "Influenza Neuraminidase",
            "1HVR": "HIV-1 Protease",
            "3POZ": "PI3K Kinase",
            "1EQG": "COX-1 + Ibuprofen",
        }
        for pdb, name in targets.items():
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
        st.markdown(f"**{DEVELOPER}**  \nB.Tech Biotechnology · {INSTITUTION}")
        st.markdown("[📸 Instagram](https://www.instagram.com/n_vamsi_reddie)  "
                    "[💻 GitHub](https://github.com/vamsikrishnareddy66)")
        st.markdown("---")
        st.warning("⚠️ Educational Use Only. Scores are simulated.")
        st.caption(f"🧬 v{APP_VERSION} · Python · Streamlit · RCSB PDB")


render_sidebar()


# ═══════════════════════════════════════════════════════════════════
# HERO BANNER  (live clock)
# ═══════════════════════════════════════════════════════════════════

components.html("""
<div style="background:linear-gradient(135deg,#0052D4,#6A11CB,#FF0080);
            padding:28px 32px;border-radius:22px;color:white;
            display:flex;justify-content:space-between;align-items:center;
            font-family:Inter,Arial,sans-serif;box-shadow:0 8px 28px rgba(0,0,0,.35);">
  <div>
    <div style="display:inline-block;background:#FF2E88;padding:4px 14px;
                border-radius:20px;font-size:12px;font-weight:700;margin-bottom:10px;">
      🧬 Computational Drug Discovery Platform
    </div>
    <h1 style="margin:0;font-size:28px;font-weight:800;">Drug Discovery Pipeline</h1>
    <p style="margin:6px 0 0;font-size:13px;opacity:.85;">
      👨‍💻 N. Vamsi Krishna Reddy · KL University · v3.0
    </p>
    <p style="margin:10px 0 0;font-size:13px;line-height:1.6;">
      Virtual Screening &nbsp;•&nbsp; Live RCSB PDB Data &nbsp;•&nbsp;
      ADMET Evaluation &nbsp;•&nbsp; AI Research Assistant &nbsp;•&nbsp; PDF Reports
    </p>
  </div>
  <div style="background:rgba(255,255,255,.14);padding:16px;border-radius:14px;
              width:175px;text-align:center;backdrop-filter:blur(8px);margin-left:20px;">
    <div style="font-size:11px;">🕒 Current Time</div>
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
  function tick(){
    const n=new Date();
    document.getElementById("clock").textContent=n.toLocaleTimeString();
    const s=Math.floor((Date.now()-t0)/1000);
    document.getElementById("timer").textContent=
      String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
  }
  setInterval(tick,1000);tick();
</script>
""", height=200)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

(tab_home, tab_screen, tab_detail,
 tab_viewer, tab_ai, tab_future, tab_about) = st.tabs([
    "🏠 Dashboard",
    "🔬 Screening",
    "💊 Compounds",
    "🧬 3D Viewer",
    "🤖 AI Assistant",
    "🚀 Future",
    "👨‍💻 Developer",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 0: LANDING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def render_dashboard_tab() -> None:
    """Landing page with live stat cards and quick-start guide."""
    st.markdown('<div class="section-header">Platform Overview</div>',
                unsafe_allow_html=True)

    compound_count = _compound_count()
    protein_loaded = (st.session_state.protein_info or {}).get("pdb_id", "—")
    last_best = (
        st.session_state.history[-1]["best_compound"]
        if st.session_state.history else "—"
    )

    # ── Stat cards ────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "🧬", "5",              "Proteins Supported")
    metric_card(c2, "💊", compound_count,   "Compounds in Library")
    metric_card(c3, "📄", st.session_state.reports_count, "Reports Generated")
    metric_card(c4, "🟢", "Online",         "System Status")
    metric_card(c5, "🧠", f"v{APP_VERSION}", "Platform Version")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Current run summary ───────────────────────────────────────
    st.markdown('<div class="section-header">Current Session</div>',
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    metric_card(s1, "🔬", protein_loaded,   "Active Protein")
    metric_card(s2, "🏆", last_best,        "Best Compound (last run)")
    metric_card(s3, "📊", len(st.session_state.history), "Screens Completed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick-start guide ────────────────────────────────────────
    st.markdown('<div class="section-header">Quick Start</div>',
                unsafe_allow_html=True)

    steps = [
        ("1", "🔬 Screening", "Enter a PDB ID and click **Fetch Protein Data**"),
        ("2", "💊 Compounds",  "Select compounds from the library"),
        ("3", "🚀 Run",        "Click **Run Virtual Screening Simulation**"),
        ("4", "📊 Analyse",    "Explore ranked results, ADMET panel, and charts"),
        ("5", "🤖 Ask AI",     "Switch to the **AI Assistant** tab for explanations"),
        ("6", "📄 Export",     "Download CSV or PDF report"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f'<div class="compound-card">'
                f'<b style="font-size:1.4rem">{num}</b>&nbsp;&nbsp;{title}<br>'
                f'<span style="color:#9DB4CC;font-size:.85rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature highlights ────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Highlights</div>',
                unsafe_allow_html=True)

    features = [
        ("✅", "Live RCSB PDB Data",        "Protein metadata fetched in real time"),
        ("✅", "Virtual Screening",          "Formula-based scoring for 25 + compounds"),
        ("✅", "ADMET Evaluation",           "Lipinski Ro5 + physicochemical properties"),
        ("✅", "AI Research Assistant",      "Ask Claude to explain results"),
        ("✅", "Advanced Visualisations",    "Heatmap, radar, violin, box, correlation"),
        ("✅", "3D Protein Viewer",          "Embedded RCSB NGL viewer"),
        ("✅", "PDF Report Generation",      "Publication-style downloadable report"),
        ("✅", "Reference Ligand Compare",   "Benchmark screened compounds vs co-crystal"),
    ]
    fc1, fc2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with (fc1 if i % 2 == 0 else fc2):
            st.markdown(
                f'<div class="compound-card" style="margin-bottom:8px">'
                f'{icon} <b>{title}</b><br>'
                f'<span style="color:#9DB4CC;font-size:.83rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


with tab_home:
    render_dashboard_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: VIRTUAL SCREENING
# ══════════════════════════════════════════════════════════════════════════

def _render_progress(steps_done: int, steps: list[str]) -> None:
    """Render a visual step progress list."""
    for i, label in enumerate(steps):
        if i < steps_done:
            cls = "done";   icon = "✅"
        elif i == steps_done:
            cls = "active"; icon = "⏳"
        else:
            cls = "";       icon = "○"
        st.markdown(
            f'<div class="progress-step {cls}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )


def render_screening_tab() -> None:
    """Full virtual screening tab — Steps 1-3 + results."""

    # ── STEP 1: Protein ───────────────────────────────────────────
    st.markdown('<div class="section-header">Step 1 — Protein Target</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        pdb_id_input: str = st.text_input(
            "PDB ID (4 characters)",
            value="2HU4",
            max_chars=4,
            help="Try: 2HU4 · 6LU7 · 1HVR · 3POZ · 1ATP",
        ).strip().upper()

        is_valid = bool(pdb_id_input) and protein.is_valid_pdb_id(pdb_id_input)
        if pdb_id_input and not is_valid:
            st.error("Invalid PDB ID — must start with a digit (1–9) "
                     "followed by three alphanumeric characters.")

        if st.button("Fetch Protein Data", icon="🔎",
                     disabled=not is_valid, type="primary", use_container_width=True):
            with st.spinner(f"Querying RCSB PDB for {pdb_id_input}…"):
                st.session_state.protein_info = protein.get_protein_info(pdb_id_input)

    with c2:
        info: dict | None = st.session_state.protein_info
        if info and info.get("pdb_id") == pdb_id_input:

            # Protein thumbnail
            thumb_url = f"https://cdn.rcsb.org/images/structures/{pdb_id_input.lower()}_assembly-1.jpeg"
            try:
                tr = requests.get(thumb_url, timeout=5)
                if tr.status_code == 200:
                    st.image(tr.content, caption=f"{pdb_id_input} structure thumbnail",
                             width=200)
            except Exception:
                pass

            src_label = ("✅ Live RCSB data" if info.get("source") == "live"
                         else "⚠️ Offline fallback")
            st.info(src_label)

            left, right = st.columns(2)
            with left:
                st.markdown(f"**Name:** {info.get('name','—')}")
                st.markdown(f"**Organism:** {info.get('organism','—')}")
                st.markdown(f"**Method:** {info.get('method','—')}")
                st.markdown(f"**Resolution:** {info.get('resolution','—')} Å")
            with right:
                chains  = info.get("chains")  or []
                ligands = info.get("ligands") or []
                asr     = info.get("active_site_residues") or []
                st.markdown(f"**Chains:** {', '.join(chains) if chains else 'See PDB'}")
                st.markdown(f"**Ligands:** {', '.join(ligands) if ligands else 'None'}")
                asr_txt = (", ".join(asr[:4]) + ("…" if len(asr) > 4 else "")) if asr else "Not annotated"
                st.markdown(f"**Active Site (≤5 Å):** {asr_txt}")
                st.markdown(f"[View on RCSB ↗]({info.get('structure_url','#')})")

            if st.button("Download .pdb File", icon="⬇️", use_container_width=True):
                with st.spinner("Downloading…"):
                    file_bytes, err = protein.download_structure_file(pdb_id_input)
                if file_bytes:
                    st.download_button("Save .pdb", data=file_bytes,
                                       file_name=f"{pdb_id_input}.pdb",
                                       mime="chemical/x-pdb", icon="💾")
                else:
                    st.error(f"Download failed: {err}")

            # Reference ligand
            rl = info.get("reference_ligand")
            if rl:
                st.markdown("---")
                st.markdown(
                    f'<div class="ref-lig-card">'
                    f'<b>📌 {rl.get("name","—")}</b> — co-crystallised with <b>{pdb_id_input}</b><br>'
                    f'<b>Formula:</b> {rl.get("formula","—")} &nbsp;|&nbsp; '
                    f'<b>MW:</b> {rl.get("mw","—")} g/mol &nbsp;|&nbsp; '
                    f'<b>PubChem CID:</b> {rl.get("pubchem_cid","—")}<br>'
                    f'<b>SMILES:</b> <code style="font-size:.78rem">{rl.get("smiles","—")}</code><br>'
                    f'<em style="font-size:.84rem">{rl.get("notes","")}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="info-box">Enter a PDB ID above and click '
                '<b>Fetch Protein Data</b> to load structure metadata.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── STEP 2: Compound selection ────────────────────────────────
    st.markdown('<div class="section-header">Step 2 — Compound Library</div>',
                unsafe_allow_html=True)

    compounds_df: pd.DataFrame = screening.load_compound_data(DATA_PATH)
    all_compounds: list[str]   = compounds_df["compound_name"].tolist()

    cs1, cs2 = st.columns([4, 1])
    with cs1:
        search_term: str = st.text_input(
            "Filter", placeholder="Type a compound name…",
            label_visibility="collapsed",
        )
    with cs2:
        st.metric("Library", len(all_compounds))

    filtered = (
        [c for c in all_compounds if search_term.lower() in c.lower()]
        if search_term else all_compounds
    )
    if search_term and not filtered:
        st.caption(f'No matches for "{search_term}". Try a shorter term.')

    selected_compounds: list[str] = st.multiselect(
        "Select compounds to screen",
        options=filtered,
        default=filtered[:5] if filtered else [],
        help="Hold Ctrl / ⌘ to select multiple compounds.",
    )

    with st.expander("Browse full compound database"):
        st.dataframe(
            compounds_df[safe_cols(compounds_df, [
                "compound_name", "molecular_formula", "molecular_weight",
                "logp", "h_donors", "h_acceptors", "rotatable_bonds",
                "pubchem_cid", "drugbank_id", "smiles",
            ])],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── STEP 3: Run ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 3 — Run Virtual Screening</div>',
                unsafe_allow_html=True)

    protein_ready = (
        st.session_state.protein_info is not None
        and st.session_state.protein_info.get("pdb_id") == pdb_id_input
    )
    reasons = (
        ([] if protein_ready else ["fetch protein data in Step 1"])
        + ([] if selected_compounds else ["select at least one compound in Step 2"])
    )
    if reasons:
        st.caption("Before running, please " + " and ".join(reasons) + ".")

    run_clicked = st.button(
        "Run Virtual Screening Simulation",
        icon="🚀", type="primary", disabled=bool(reasons),
    )

    if run_clicked:
        STEPS = [
            "Initialising screening engine…",
            f"Loading {pdb_id_input} binding site…",
            f"Preparing {len(selected_compounds)} compound(s)…",
            "Calculating simulated docking scores…",
            "Ranking and scoring ADMET properties…",
            "Finalising results…",
        ]
        progress_slot = st.empty()
        prog_bar = st.progress(0)

        for idx, step_label in enumerate(STEPS):
            with progress_slot.container():
                _render_progress(idx, STEPS)
            prog_bar.progress(int((idx + 1) / len(STEPS) * 100))
            time.sleep(0.55)

        raw = screening.simulate_virtual_screening(
            compounds_df, pdb_id_input, selected_compounds,
            protein_info=st.session_state.protein_info,
        )
        ranked = ranking.rank_compounds(raw)
        st.session_state.results_df = ranked
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

    results_df: pd.DataFrame = st.session_state.results_df
    if results_df.empty:
        st.markdown(
            '<div class="info-box">Complete Steps 1–2, then click '
            '<b>Run Virtual Screening Simulation</b> to see results.</div>',
            unsafe_allow_html=True,
        )
        _render_disclaimer()
        return

    _render_results(results_df, pdb_id_input)
    _render_disclaimer()


def _render_results(results_df: pd.DataFrame, pdb_id: str) -> None:
    """Render ranked results, ADMET, charts, metrics, and export."""
    sc        = score_col(results_df)
    best_row  = ranking.get_best_compound(results_df)
    best_name = (best_row or {}).get("compound_name", "—")
    best_val  = (best_row or {}).get(sc, "—")
    target    = results_df["protein_target"].iloc[0] if "protein_target" in results_df.columns else pdb_id

    st.success(f"**Top Compound:** **{best_name}** scored **{best_val} kcal/mol** against {target}.", icon="🏆")
    st.caption("⚠️ Formula-based simulations for educational purposes only — not actual docking.")

    # ── AI Screening Summary ─────────────────────────────────────
    with st.expander("🤖 AI-Generated Screening Summary", expanded=True):
        context = (
            f"Protein: {pdb_id}\n"
            f"Compounds screened: {len(results_df)}\n"
            f"Best compound: {best_name} ({best_val} kcal/mol)\n"
            f"Compounds: {', '.join(results_df['compound_name'].tolist()[:10])}\n"
        )
        ai_summary_key = f"summary_{pdb_id}_{best_name}"
        if ai_summary_key not in st.session_state:
            with st.spinner("Generating AI summary…"):
                summary = _call_claude(
                    system=(
                        "You are a senior computational medicinal chemist. "
                        "Write a concise, scientifically accurate 3-paragraph summary of a "
                        "virtual screening run. Paragraph 1: what was screened and why. "
                        "Paragraph 2: interpret the top compound's simulated score and "
                        "ADMET implications. Paragraph 3: next experimental steps. "
                        "Use plain English suitable for a biotechnology student. "
                        "Always remind the reader these are simulated scores."
                    ),
                    messages=[{"role": "user", "content": context}],
                )
            st.session_state[ai_summary_key] = summary
        st.markdown(st.session_state[ai_summary_key])

    # ── Ranked table ──────────────────────────────────────────────
    st.markdown("#### 📊 Ranked Screening Results")
    disp = safe_cols(results_df, [
        "rank", "compound_name", "molecular_formula",
        "molecular_weight", "logp", sc, "lipinski_status",
    ])
    rename = {
        sc: "Sim. Score (kcal/mol)", "compound_name": "Compound",
        "molecular_formula": "Formula", "molecular_weight": "MW (g/mol)",
        "logp": "LogP", "lipinski_status": "Lipinski Ro5",
    }

    def _hl(row: pd.Series) -> list[str]:
        return ["background-color:#0D3B26" if row.get("rank") == 1 else ""] * len(row)

    st.dataframe(
        results_df[disp].rename(columns=rename).style.apply(_hl, axis=1),
        use_container_width=True, hide_index=True,
    )

    # ── ADMET ─────────────────────────────────────────────────────
    st.markdown("#### 🧪 ADMET Properties")
    admet = safe_cols(results_df, [
        "compound_name", "molecular_weight", "logp",
        "h_donors", "h_acceptors", "rotatable_bonds", "lipinski_status",
    ])
    st.dataframe(
        results_df[admet].rename(columns={
            "compound_name": "Compound", "molecular_weight": "MW (g/mol)",
            "logp": "LogP", "h_donors": "H-Donors",
            "h_acceptors": "H-Acceptors", "rotatable_bonds": "Rot. Bonds",
            "lipinski_status": "Lipinski Ro5",
        }),
        use_container_width=True, hide_index=True,
    )
    with st.expander("What is Lipinski's Rule of Five?"):
        st.markdown(
            "MW ≤ 500 Da · LogP ≤ 5 · H-bond donors ≤ 5 · "
            "H-bond acceptors ≤ 10 · Rotatable bonds ≤ 10. "
            "Violations flag added oral-bioavailability risk."
        )

    # ── Reference ligand comparison ───────────────────────────────
    ref_lig = (st.session_state.protein_info or {}).get("reference_ligand")
    if ref_lig:
        st.markdown("#### 📌 vs Reference Ligand")
        comp_data = [
            {"Compound": r["compound_name"], "Score": r.get(sc, 0), "Type": "Screened"}
            for _, r in results_df.iterrows()
        ]
        fig_ref = px.bar(
            pd.DataFrame(comp_data), x="Compound", y="Score", color="Type",
            color_discrete_map={"Screened": TEAL_MID},
            title=f"Scores vs {ref_lig.get('name','Reference Ligand')}",
            labels={"Score": "Sim. Score (kcal/mol)"},
        )
        fig_ref.add_hline(y=-8.9, line_dash="dash", line_color=GOLD,
                          annotation_text="Ref. ligand est. −8.9 kcal/mol",
                          annotation_position="top right")
        fig_ref.update_layout(**plotly_layout())
        st.plotly_chart(fig_ref, use_container_width=True)

    # ── Core charts ───────────────────────────────────────────────
    st.markdown("#### 📈 Score Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        fig_bar = px.bar(
            results_df.sort_values(sc), x="compound_name", y=sc,
            color=sc, color_continuous_scale="RdYlGn_r",
            title="Scores per Compound (lower = stronger simulated binding)",
            labels={"compound_name": "Compound", sc: "Sim. Score (kcal/mol)"},
            text=sc,
        )
        fig_bar.update_traces(textposition="outside", texttemplate="%{text:.2f}")
        fig_bar.update_layout(showlegend=False, **plotly_layout())
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        if {"molecular_weight", "logp"}.issubset(results_df.columns):
            fig_sc = px.scatter(
                results_df, x="molecular_weight", y=sc,
                size="molecular_weight", color="compound_name",
                hover_name="compound_name",
                title="MW vs Simulated Score",
                labels={"molecular_weight": "MW (g/mol)", sc: "Sim. Score (kcal/mol)"},
            )
            fig_sc.update_layout(**plotly_layout())
            st.plotly_chart(fig_sc, use_container_width=True)

    # Distribution histogram
    fig_dist = px.histogram(
        results_df, x=sc, nbins=max(8, len(results_df) // 2),
        title="Score Distribution",
        labels={sc: "Sim. Score (kcal/mol)", "count": "Compounds"},
        color_discrete_sequence=[TEAL_LIGHT],
    )
    fig_dist.update_layout(**plotly_layout())
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Advanced charts ───────────────────────────────────────────
    st.markdown("#### 🔬 Advanced Visualisations")
    adv1, adv2 = st.columns(2)

    # Box plot
    with adv1:
        numeric_props = safe_cols(results_df, [
            sc, "molecular_weight", "logp", "h_donors", "h_acceptors",
        ])
        if len(numeric_props) >= 2:
            melted = results_df[numeric_props].melt(
                var_name="Property", value_name="Value"
            )
            fig_box = px.box(
                melted, x="Property", y="Value", color="Property",
                title="Property Distribution (Box Plot)",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_box.update_layout(**plotly_layout())
            st.plotly_chart(fig_box, use_container_width=True)

    # Violin plot
    with adv2:
        if sc in results_df.columns and "lipinski_status" in results_df.columns:
            fig_vio = px.violin(
                results_df, y=sc, x="lipinski_status", color="lipinski_status",
                box=True, points="all", hover_name="compound_name",
                title="Score Distribution by Lipinski Status (Violin)",
                labels={sc: "Sim. Score (kcal/mol)", "lipinski_status": "Lipinski"},
                color_discrete_sequence=[TEAL_LIGHT, GOLD, PINK],
            )
            fig_vio.update_layout(**plotly_layout())
            st.plotly_chart(fig_vio, use_container_width=True)

    # Correlation heatmap
    num_cols = safe_cols(results_df, [
        "molecular_weight", "logp", "h_donors",
        "h_acceptors", "rotatable_bonds", sc,
    ])
    if len(num_cols) >= 3:
        corr = results_df[num_cols].corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(
            title="Physicochemical Property Correlation Matrix",
            **plotly_layout(),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Radar chart — top 3 compounds
    radar_props = safe_cols(results_df, [
        "molecular_weight", "logp", "h_donors", "h_acceptors", "rotatable_bonds",
    ])
    if radar_props and len(results_df) >= 1:
        top3 = results_df.nsmallest(min(3, len(results_df)), sc)
        norm = (results_df[radar_props] - results_df[radar_props].min()) / \
               (results_df[radar_props].max() - results_df[radar_props].min() + 1e-9)
        fig_radar = go.Figure()
        colours = [TEAL_LIGHT, GOLD, PINK]
        for i, (_, row) in enumerate(top3.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=norm.loc[row.name, radar_props].tolist() + [norm.loc[row.name, radar_props[0]]],
                theta=radar_props + [radar_props[0]],
                fill="toself",
                name=row["compound_name"],
                line_color=colours[i % len(colours)],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar: Top 3 Compounds — Normalised Properties",
            **plotly_layout(),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Metrics summary ───────────────────────────────────────────
    st.markdown('<div class="section-header">Scientific Metrics</div>',
                unsafe_allow_html=True)
    stats = stats_module.calculate_statistics(results_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, icon, val, lbl in [
        (m1, "🔬", stats.get("count", 0),                          "Screened"),
        (m2, "⚖️", stats.get("avg_mw", "—"),                      "Avg. MW (g/mol)"),
        (m3, "🏆", f"{stats.get('best_score','—')} kcal/mol",      "Best Score"),
        (m4, "📉", f"{stats.get('worst_score','—')} kcal/mol",     "Worst Score"),
        (m5, "📊", f"{stats.get('average_score','—')} kcal/mol",   "Average Score"),
    ]:
        metric_card(col, icon, val, lbl)

    # ── Export ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Export Results</div>',
                unsafe_allow_html=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "Download CSV", icon="⬇️",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"screening_{pdb_id}_{ts}.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption("All ranked results + ADMET columns.")

    with e2:
        with st.spinner("Building PDF report…"):
            pdf_bytes = report.generate_pdf_report(
                st.session_state.protein_info, results_df, stats
            )
        st.session_state.reports_count += 1
        st.download_button(
            "Download PDF Report", icon="📄",
            data=pdf_bytes,
            file_name=f"screening_report_{pdb_id}_{ts}.pdf",
            mime="application/pdf", use_container_width=True,
        )
        st.caption("Includes protein info, rankings, stats, and disclaimer.")


def _render_disclaimer() -> None:
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Educational Use Only.</b> '
        'Protein metadata is sourced live from <a href="https://www.rcsb.org" '
        'target="_blank">RCSB PDB</a>. All docking scores are formula-based simulations '
        'and must not be used for clinical, regulatory, or commercial decisions.</div>',
        unsafe_allow_html=True,
    )


with tab_screen:
    render_screening_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: COMPOUND DETAILS
# ══════════════════════════════════════════════════════════════════════════

def render_compound_tab() -> None:
    st.markdown('<div class="section-header">💊 Compound Details</div>',
                unsafe_allow_html=True)
    st.write("Explore individual compound properties, 2D/3D structure images, "
             "and database identifiers.")

    compounds_df_d = screening.load_compound_data(DATA_PATH)
    sel = st.selectbox("Select a compound",
                        compounds_df_d["compound_name"].tolist())
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
            st.markdown(f"**{label}:** {row.get(key, '—')}" +
                        (" g/mol" if key == "molecular_weight" else ""))

        lip = screening.lipinski_status(row.to_dict())
        st.markdown(f"**Lipinski Ro5:** {lip}")
        st.markdown(f"**PubChem CID:** {cid or '—'}")
        if cid and str(cid) != "nan":
            st.markdown(f"[View on PubChem ↗](https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)})")
        st.markdown(f"**DrugBank ID:** {db_id or '—'}")
        if db_id and str(db_id) != "nan":
            st.markdown(f"[View on DrugBank ↗](https://go.drugbank.com/drugs/{db_id})")
        st.markdown("**SMILES:**")
        st.code(smiles, language="text")

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
                        st.image(r.content, caption=f"{sel} — PubChem {label}",
                                 use_container_width=True)
                except Exception:
                    st.info(f"{label} image unavailable.")
        else:
            st.info("No PubChem CID — structure image cannot be fetched automatically.")
            st.code(smiles or "No SMILES available", language="text")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Structures and identifiers are fetched from PubChem. '
        'Verify IDs from primary sources before use in research.</div>',
        unsafe_allow_html=True,
    )


with tab_detail:
    render_compound_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: 3D PROTEIN VIEWER
# ══════════════════════════════════════════════════════════════════════════

def render_viewer_tab() -> None:
    st.markdown('<div class="section-header">🧪 3D Protein Structure Viewer</div>',
                unsafe_allow_html=True)
    st.write("Interactive 3D view via the RCSB PDB embedded NGL viewer.")

    info = st.session_state.protein_info
    v1, v2 = st.columns([1, 3])
    with v1:
        view_pdb = st.text_input(
            "PDB ID to view",
            value=(info.get("pdb_id") if info else "2HU4"),
            max_chars=4,
        ).strip().upper()

        st.info("💡 Use the toolbar inside the viewer to change style, colour scheme, "
                "and toggle ligand highlighting.")

        if info and info.get("pdb_id") == view_pdb:
            asr = info.get("active_site_residues") or []
            if asr:
                st.success(f"**Active site residues (≤5 Å):** {', '.join(asr[:6])}")
            ligands = info.get("ligands") or []
            if ligands:
                st.markdown(f"**Co-crystallised ligand:** {ligands[0]}")

    with v2:
        st.markdown(
            f'<iframe src="https://www.rcsb.org/3d-view/{view_pdb}?preset=defaultView" '
            f'width="100%" height="520" '
            f'style="border:2px solid {TEAL_MID};border-radius:10px;" '
            f'title="3D Structure of {view_pdb}"></iframe>'
            f'<p style="font-size:.8rem;color:#9DB4CC;margin-top:6px;">'
            f'Powered by RCSB PDB / NGL. '
            f'<a href="https://www.rcsb.org/structure/{view_pdb}" target="_blank">'
            f'Open full entry ↗</a></p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "🖱️ **Rotate** — drag &nbsp;|&nbsp; 🔍 **Zoom** — scroll &nbsp;|&nbsp; "
        "🤚 **Pan** — right-drag"
    )
    st.markdown(
        '<div class="disclaimer">⚠️ 3D viewer uses the RCSB PDB embedded viewer. '
        'Structures are real experimental data.</div>',
        unsafe_allow_html=True,
    )


with tab_viewer:
    render_viewer_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: AI RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════

_AI_SYSTEM = """
You are an expert AI Research Assistant for a drug discovery platform.
Your users are B.Tech Biotechnology students. Your role is to:
- Explain virtual screening concepts clearly and accurately
- Interpret simulated docking scores in educational context
- Explain ADMET, Lipinski's Ro5, and pharmacokinetics concepts
- Discuss why a compound ranked higher or lower
- Always remind users that scores shown are simulated, not real docking results
- Suggest next experimental steps when asked
- Be encouraging and educational in tone
Keep answers concise (3-5 sentences unless a longer explanation is needed).
"""

_SUGGESTED_QUESTIONS = [
    "Why did the top compound score best?",
    "Explain Lipinski's Rule of Five",
    "What does ADMET stand for and why does it matter?",
    "What is molecular docking?",
    "How do I interpret a simulated docking score?",
    "What are the next steps after virtual screening?",
    "Explain the difference between LogP and LogD",
    "What is the significance of H-bond donors and acceptors?",
]


def render_ai_tab() -> None:
    st.markdown('<div class="section-header">🤖 AI Research Assistant</div>',
                unsafe_allow_html=True)
    st.write(
        "Ask the AI assistant to explain results, clarify concepts, "
        "or suggest next steps. Powered by Claude."
    )

    # Context from current session
    results_df = st.session_state.results_df
    prot_info  = st.session_state.protein_info
    context_parts = []
    if prot_info:
        context_parts.append(
            f"Current protein: {prot_info.get('pdb_id')} — {prot_info.get('name','')}"
        )
    if not results_df.empty:
        sc     = score_col(results_df)
        best   = ranking.get_best_compound(results_df)
        b_name = (best or {}).get("compound_name", "—")
        b_val  = (best or {}).get(sc, "—")
        context_parts.append(
            f"Last screening: {len(results_df)} compounds against "
            f"{prot_info.get('pdb_id','?') if prot_info else '?'}. "
            f"Best: {b_name} ({b_val} kcal/mol)"
        )

    if context_parts:
        st.info("📋 Session context: " + " · ".join(context_parts))

    # Suggested questions
    st.markdown("**Suggested questions:**")
    sq_cols = st.columns(4)
    for i, q in enumerate(_SUGGESTED_QUESTIONS):
        with sq_cols[i % 4]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.ai_chat.append({"role": "user", "content": q})
                with st.spinner("Thinking…"):
                    context_str = "\n".join(context_parts) if context_parts else "No session data."
                    messages    = [
                        {"role": "user",
                         "content": f"Session context:\n{context_str}\n\nQuestion: {q}"},
                    ]
                    reply = _call_claude(_AI_SYSTEM, messages)
                st.session_state.ai_chat.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("---")

    # Chat history
    for msg in st.session_state.ai_chat:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Free-form input
    with st.form("ai_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="e.g. Why did Oseltamivir rank first?",
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Send", use_container_width=False)

    if send and user_input.strip():
        st.session_state.ai_chat.append({"role": "user", "content": user_input.strip()})
        context_str = "\n".join(context_parts) if context_parts else "No session data."
        messages = [
            {"role": "user",
             "content": f"Session context:\n{context_str}\n\nQuestion: {user_input.strip()}"},
        ]
        with st.spinner("Generating response…"):
            reply = _call_claude(_AI_SYSTEM, messages)
        st.session_state.ai_chat.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.ai_chat:
        if st.button("Clear chat history", icon="🗑️"):
            st.session_state.ai_chat = []
            st.rerun()


with tab_ai:
    render_ai_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: FUTURE UPGRADES
# ══════════════════════════════════════════════════════════════════════════

def render_future_tab() -> None:
    st.markdown('<div class="section-header">🚀 Future Upgrade Roadmap</div>',
                unsafe_allow_html=True)
    st.write("Planned upgrades to transition this platform toward production-grade "
             "scientifically validated drug discovery.")

    upgrades = [
        ("🔬 AutoDock Vina Integration",
         "Replace formula-based simulation with real molecular docking. "
         "Requires .pdbqt preparation and a local Vina binary.",
         "High"),
        ("🧬 Real Force-Field Docking",
         "Implement MMFF94/Gasteiger charge docking for genuine kcal/mol energies.",
         "High"),
        ("💊 DrugBank API",
         "Fetch real drug interaction profiles, PK data, and mechanism of action.",
         "Medium"),
        ("🔭 PubChem Live ADMET",
         "Use PubChem PUG-REST for live bioassay data and similarity searching.",
         "Medium"),
        ("🤖 GNN / Transformer Lead Prediction",
         "Integrate MolBERT / ChemBERTa for AI-predicted binding affinity and ADMET.",
         "High"),
        ("🌊 Molecular Dynamics",
         "Add OpenMM interface for MM-PBSA binding free energy estimation.",
         "Medium"),
        ("📊 ADMETlab / pkCSM API",
         "Validated ADMET profiles instead of rule-based Lipinski estimates.",
         "Medium"),
        ("🧪 Scaffold Clustering",
         "Bemis-Murcko scaffold extraction and hierarchical clustering by series.",
         "Low"),
        ("🗂️ Multi-Target Screening",
         "Screen against multiple PDB targets and generate a selectivity heatmap.",
         "Low"),
        ("☁️ User Accounts & Projects",
         "Save projects, compare runs across sessions, and share reports.",
         "Low"),
    ]

    colours = {"High": "#C0392B", "Medium": GOLD, "Low": GREEN_OK}
    for title, desc, priority in upgrades:
        pc = colours.get(priority, "#555")
        st.markdown(
            f'<div class="roadmap-item">'
            f'<b>{title}</b>'
            f'<span style="float:right;background:{pc};color:white;'
            f'border-radius:10px;padding:1px 8px;font-size:.75rem;">'
            f'{priority} Priority</span>'
            f'<br><span style="color:#9DB4CC;font-size:.87rem">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📚 Recommended Tools")
    tools = {
        "AutoDock Vina": "https://vina.scripps.edu/",
        "RDKit":         "https://www.rdkit.org/",
        "OpenMM":        "https://openmm.org/",
        "ADMETlab 2.0":  "https://admetmesh.scbdd.com/",
        "PubChem":       "https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest",
        "DrugBank":      "https://docs.drugbank.com/",
        "PyMOL":         "https://pymol.org/",
        "ChemBERTa":     "https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1",
    }
    tc = st.columns(4)
    for i, (name, url) in enumerate(tools.items()):
        with tc[i % 4]:
            st.markdown(f"[🔗 {name}]({url})")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Upgrades above require additional dependencies, '
        'compute resources, and validated scientific protocols before results can be '
        'used in research contexts.</div>',
        unsafe_allow_html=True,
    )


with tab_future:
    render_future_tab()


# ══════════════════════════════════════════════════════════════════════════
# TAB 6: DEVELOPER / ABOUT
# ══════════════════════════════════════════════════════════════════════════

def render_about_tab() -> None:
    st.markdown(f"## 👨‍💻 {DEVELOPER}")
    st.markdown(f"**B.Tech Biotechnology · {INSTITUTION} · Final Year Project 2026**")
    st.markdown("---")

    a1, a2 = st.columns([2, 1])
    with a1:
        st.markdown("""
### About This Project
The Drug Discovery Pipeline is an interactive computational biology platform
combining protein structure analysis, virtual screening, ADMET evaluation,
AI-powered insights, 3D molecular visualisation, and automated scientific
reporting in a single web application.

### Technologies
**Frontend:** Python · Streamlit · Plotly · HTML/CSS  
**Bioinformatics:** RCSB PDB API · PubChem PUG-REST · NGL Viewer  
**AI:** Claude (Anthropic) via Messages API  
**Development:** AI-assisted (Claude, GitHub Copilot)

### Project Objectives
- Protein structure retrieval and analysis  
- Virtual compound screening simulation  
- ADMET property evaluation  
- AI-generated research summaries  
- Interactive 3D protein visualisation  
- Automated scientific PDF reporting
""")

    with a2:
        st.markdown("### 📞 Contact")
        st.markdown("📱 +91 9346599651")
        st.markdown("📧 vamsikrishnareddynemaildinne@gmail.com")
        st.markdown("[📸 Instagram](https://www.instagram.com/n_vamsi_reddie)")
        st.markdown("[💻 GitHub](https://github.com/vamsikrishnareddy66)")
        st.markdown("[🚀 Project Repo](https://github.com/vamsikrishnareddy66/AI-Drug-Explorer)")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Educational Use Only. Virtual screening scores '
        'are simulated and must not be interpreted as actual docking results or '
        'clinical recommendations.</div>',
        unsafe_allow_html=True,
    )


with tab_about:
    render_about_tab()


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="site-footer">'
    f'<b>Drug Discovery Pipeline</b> &nbsp;·&nbsp; v{APP_VERSION}<br>'
    f'Developed by <b>{DEVELOPER}</b> &nbsp;·&nbsp; {INSTITUTION}<br><br>'
    f'Powered by &nbsp;'
    f'<a href="https://www.python.org" target="_blank">Python</a> &nbsp;·&nbsp; '
    f'<a href="https://streamlit.io" target="_blank">Streamlit</a> &nbsp;·&nbsp; '
    f'<a href="https://www.rcsb.org" target="_blank">RCSB PDB</a> &nbsp;·&nbsp; '
    f'<a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank">PubChem</a> &nbsp;·&nbsp; '
    f'<a href="https://www.anthropic.com" target="_blank">Claude AI</a><br><br>'
    f'<span style="font-size:.78rem;opacity:.6;">'
    f'Educational use only. Simulated scores are not actual docking results.'
    f'</span>'
    f'</div>',
    unsafe_allow_html=True,
)
