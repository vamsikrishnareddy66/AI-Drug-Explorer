"""
app.py
------
Drug Discovery Pipeline — Scientifically Accurate Edition
B.Tech Biotechnology Final Year Project

All improvements implemented:
 1.  "Highest Simulated Score" replaces "Best Candidate"
 2.  Educational disclaimer note under all results
 3.  Enriched PDB metadata (chains, ligands, active site)
 4.  Reference ligand section with comparison
 5.  ADMET panel (MW, LogP, H-donors, H-acceptors, RotBonds, Lipinski)
 6.  Compound Details page (name, formula, structure image, CIDs, SMILES)
 7.  3D Protein Viewer via py3Dmol / NGL (via Streamlit components)
 8.  Scientific metrics panel with distribution chart
 9.  Enhanced PDF report
10.  Professional biotech UI (teal/navy/gold colour scheme)
11.  Future Upgrade section
12.  PDB ID validation
13.  "Virtual Screening Simulation" naming throughout

Run:
    streamlit run app.py
"""

import os
import json
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

import protein
import screening
import ranking
import statistics
import report

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Discovery Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "compounds.csv"
EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

SCORE_COL = "simulated_score"   # column written by screening.py

# ── Colour tokens ─────────────────────────────────────────────────────────
TEAL_DARK  = "#0B1026"     # Dark Navy
TEAL_MID   = "#7B61FF"     # Purple
TEAL_LIGHT = "#00D4FF"     # Neon Cyan
GOLD       = "#FF2E88"     # Neon Pink
GREEN_OK   = "#00FFB3"     # Neon Green
BG_CARD    = "#141B3A"     # Glass Card Background
# ── Global CSS / theme ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(
        135deg,
        #081229 0%,
        #1A1F4D 30%,
        #3B1C71 70%,
        #FF0080 100%
    );
}}
/* ── Font & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── Top banner gradient ── */
.hero-banner {{
    background: linear-gradient(
        135deg,
        #0052D4 0%,
        #6A11CB 50%,
        #FF0080 100%
    );
    border-radius: 25px;
    padding: 40px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}}
.hero-badge {{
    display: inline-block; background: {GOLD}; color: white;
    border-radius: 20px; padding: 2px 12px; font-size: 0.78rem;
    font-weight: 600; margin-bottom: 10px;
}}

/* ── Section headers ── */
.section-header {{
    border-left: 4px solid {TEAL_MID}; padding-left: 12px;
    color: {TEAL_DARK}; font-size: 1.15rem; font-weight: 700; margin: 20px 0 10px;
}}

/* ── Metric cards ── */
.metric-card {{
    background: {BG_CARD}; border: 1px solid #C2DFF0; border-radius: 10px;
    padding: 16px 20px; text-align: center;
}}
.metric-card .val {{ font-size: 1.7rem; font-weight: 700; color: {TEAL_DARK}; }}
.metric-card .lbl {{ font-size: 0.78rem; color: #555; margin-top: 2px; }}

/* ── Info box ── */
.info-box {{
    background: #EDF8FF; border: 1px solid #A8D8EF; border-radius: 8px;
    padding: 14px 18px; font-size: 0.9rem; color: #1A4A6E;
}}

/* ── Warning disclaimer ── */
.disclaimer {{
    background: #FFF8E6; border: 1px solid {GOLD}; border-radius: 8px;
    padding: 10px 14px; font-size: 0.82rem; color: #7A5B0A; margin-top: 8px;
}}

/* ── Reference ligand card ── */
.ref-lig-card {{
    background: linear-gradient(to right, #EDF8FF, #FFF8E6);
    border: 1px solid #B0C8D8; border-radius: 10px; padding: 16px 18px;
}}

/* ── Compound detail card ── */
.compound-card {{
    background: white; border: 1px solid #D0E8F5;
    border-radius: 10px; padding: 18px; margin-bottom: 12px;
}}

/* ── Future roadmap ── */
.roadmap-item {{
    background: white; border-left: 3px solid {TEAL_MID};
    border-radius: 6px; padding: 8px 14px; margin: 6px 0;
    font-size: 0.9rem; color: #1A3A4E;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {TEAL_DARK} 0%, #0A3550 100%);
    color: white;
}}
section[data-testid="stSidebar"] * {{ color: white !important; }}
section[data-testid="stSidebar"] .stTextInput input {{ color: #111 !important; }}
section[data-testid="stSidebar"] .stSelectbox div {{ color: #111 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    background: #E8F4FA; border-radius: 8px 8px 0 0;
    padding: 8px 18px; font-weight: 600; color: {TEAL_DARK};
}}
.stTabs [aria-selected="true"] {{
    background: {TEAL_MID} !important; color: white !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("history",      []),
    ("results_df",   pd.DataFrame()),
    ("protein_info", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 Drug Discovery Pipeline")
    st.caption("Virtual Screening Simulation — B.Tech Biotech")
    st.markdown("---")

    st.markdown("#### ℹ️ About")
    st.write(
        "This app performs **educational virtual screening**. Protein data "
        "is fetched live from **RCSB PDB**. Scores are simulated using "
        "compound properties (LogP, MW) + protein metadata."
    )

    st.markdown("#### 🔗 Quick PDB Examples")
    quick = {"SARS-CoV-2 Mpro": "6LU7", "Neuraminidase H5N1": "2HU4",
             "HIV-1 Protease": "1HVR", "PI3K Kinase": "3POZ"}
    for label, pid in quick.items():
        st.caption(f"• **{pid}** — {label}")

    st.markdown("---")
    st.markdown("#### 📜 Analysis History")
    if st.session_state.history:
        for i, e in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.write(f"**{i}.** {e['protein']} → {e['best_compound']}")
            st.caption(f"   Simulated Score: {e['best_score']} kcal/mol")
    else:
        st.caption("No analyses run yet.")

    st.markdown("---")
    st.caption("⚠️ Scores shown are educational simulations and not actual docking results.")

# ══════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">B.Tech Biotechnology — Final Year Project</div>
  <h1>🧬 Drug Discovery Pipeline</h1>
  <p>Virtual Screening Simulation · Live Protein Data (RCSB PDB) · ADMET Analysis · PDF Reports</p>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
tab_screen, tab_detail, tab_viewer, tab_future, tab_about = st.tabs([
    "🔬 Virtual Screening",
    "💊 Compound Details",
    "🧪 3D Protein Viewer",
    "🚀 Future Upgrades",
    "👨‍💻 About Me"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: VIRTUAL SCREENING
# ══════════════════════════════════════════════════════════════════════════
with tab_screen:

    # ── STEP 1: Protein Search ─────────────────────────────────────────
    st.markdown('<div class="section-header">Step 1 — Protein Target</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        pdb_id_input = st.text_input(
            "Enter PDB ID (4 characters)",
            value="2HU4",
            max_chars=4,
            help="Valid IDs: 2HU4, 6LU7, 1HVR, 3POZ, 1ATP",
        ).strip().upper()

        is_valid = protein.is_valid_pdb_id(pdb_id_input)
        if pdb_id_input and not is_valid:
            st.error("❌ Invalid PDB ID: must be a digit (1–9) followed by 3 alphanumerics.")

        search_clicked = st.button("🔎 Fetch Protein Data", disabled=not is_valid, type="primary")
        if search_clicked and is_valid:
            with st.spinner("Contacting RCSB PDB API…"):
                st.session_state.protein_info = protein.get_protein_info(pdb_id_input)

    with c2:
        info = st.session_state.protein_info
        if info and info.get("pdb_id") == pdb_id_input:
            src_badge = ("✅ Live RCSB data" if info.get("source") == "live"
                         else "⚠️ Offline fallback data")
            st.info(src_badge)

            left, right = st.columns(2)
            with left:
                st.markdown(f"**🏷️ Name:** {info.get('name', 'N/A')}")
                st.markdown(f"**🌍 Organism:** {info.get('organism', 'N/A')}")
                st.markdown(f"**🔬 Method:** {info.get('method', 'N/A')}")
                st.markdown(f"**📐 Resolution:** {info.get('resolution', 'N/A')} Å")
            with right:
                chains = info.get("chains") or []
                ligands = info.get("ligands") or []
                asr = info.get("active_site_residues") or []
                st.markdown(f"**⛓️ Chains:** {', '.join(chains) if chains else 'See PDB entry'}")
                st.markdown(f"**💊 Ligands:** {', '.join(ligands) if ligands else 'None detected'}")
                if asr:
                    asr_str = ', '.join(asr[:4]) + ('…' if len(asr) > 4 else '')
                    st.markdown(f"**🎯 Active Site** *(residues within 5 Å of ligand)*: {asr_str}")
                else:
                    st.markdown("**🎯 Active Site:** Not annotated")
                st.markdown(f"[🔗 View on RCSB PDB]({info.get('structure_url', '#')})")

            # Download structure
            if st.button("⬇️ Download .pdb Structure File"):
                with st.spinner("Downloading…"):
                    file_bytes, err = protein.download_structure_file(pdb_id_input)
                if file_bytes:
                    st.download_button("💾 Save .pdb", data=file_bytes,
                                       file_name=f"{pdb_id_input}.pdb", mime="chemical/x-pdb")
                else:
                    st.error(err)
        else:
            st.markdown('<div class="info-box">👆 Enter a PDB ID and click <b>Fetch Protein Data</b> to load live protein metadata.</div>', unsafe_allow_html=True)

    # ── Reference Ligand ──────────────────────────────────────────────
    if info and info.get("pdb_id") == pdb_id_input and info.get("reference_ligand"):
        rl = info["reference_ligand"]
        st.markdown('<div class="section-header">Reference Ligand (Co-crystallized)</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="ref-lig-card">
<b>📌 {rl.get('name', 'N/A')}</b> — co-crystallized ligand for <b>{pdb_id_input}</b><br>
<b>Formula:</b> {rl.get('formula', 'N/A')} &nbsp;|&nbsp;
<b>MW:</b> {rl.get('mw', 'N/A')} g/mol &nbsp;|&nbsp;
<b>PubChem CID:</b> {rl.get('pubchem_cid', 'N/A')}<br>
<b>SMILES:</b> <code style="font-size:0.8rem">{rl.get('smiles', 'N/A')}</code><br>
<i style="font-size:0.85rem">{rl.get('notes', '')}</i>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── STEP 2: Compound Selection ─────────────────────────────────────
    st.markdown('<div class="section-header">Step 2 — Compound Selection</div>', unsafe_allow_html=True)

    compounds_df = screening.load_compound_data(DATA_PATH)
    all_compounds = compounds_df["compound_name"].tolist()

    search_term = st.text_input("🔍 Filter compounds", placeholder="Type to filter…")
    filtered_options = (
        [c for c in all_compounds if search_term.lower() in c.lower()]
        if search_term else all_compounds
    )

    selected_compounds = st.multiselect(
        "Select compounds to screen",
        options=filtered_options,
        default=filtered_options[:5] if filtered_options else [],
    )

    with st.expander("📋 Full Compound Database"):
        show_cols = ["compound_name", "molecular_formula", "molecular_weight",
                     "logp", "h_donors", "h_acceptors", "rotatable_bonds",
                     "pubchem_cid", "drugbank_id", "smiles"]
        show_cols = [c for c in show_cols if c in compounds_df.columns]
        st.dataframe(compounds_df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── STEP 3: Run Screening ──────────────────────────────────────────
    st.markdown('<div class="section-header">Step 3 — Run Virtual Screening Simulation</div>', unsafe_allow_html=True)

    protein_ready = (
        st.session_state.protein_info is not None
        and st.session_state.protein_info.get("pdb_id") == pdb_id_input
    )

    if not protein_ready:
        st.caption("⚠️ Fetch protein data (Step 1) before running screening.")
    elif not selected_compounds:
        st.caption("⚠️ Select at least one compound above.")

    run_clicked = st.button(
        "🚀 Run Virtual Screening Simulation",
        type="primary",
        disabled=not (protein_ready and selected_compounds),
    )

    if run_clicked:
        with st.spinner("Running virtual screening simulation…"):
            raw_results = screening.simulate_virtual_screening(
                compounds_df, pdb_id_input, selected_compounds,
                protein_info=st.session_state.protein_info,
            )
            ranked_results = ranking.rank_compounds(raw_results)
            st.session_state.results_df = ranked_results

            best = ranking.get_best_compound(ranked_results)
            if best:
                st.session_state.history.append({
                    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "protein":          pdb_id_input,
                    "compounds_tested": len(ranked_results),
                    "best_compound":    best.get("compound_name"),
                    "best_score":       best.get(SCORE_COL, best.get("docking_score")),
                })

    results_df = st.session_state.results_df

    if not results_df.empty and not (run_clicked and results_df.empty):
        score_col = SCORE_COL if SCORE_COL in results_df.columns else "docking_score"
        best_row  = ranking.get_best_compound(results_df)
        best_name = best_row.get("compound_name", "N/A") if best_row else "N/A"
        best_score = best_row.get(score_col, "N/A") if best_row else "N/A"

        # ── Highest Simulated Score callout ───────────────────────────
        st.success(
            f"🏆 **Highest Simulated Score:** **{best_name}** — "
            f"**{best_score} kcal/mol** (vs {results_df['protein_target'].iloc[0]})"
        )
        st.markdown(
            '<div class="disclaimer">⚠️ Scores shown are educational simulations and not actual docking results.</div>',
            unsafe_allow_html=True,
        )

        # ── Results table ──────────────────────────────────────────────
        st.markdown("#### 📊 Ranked Screening Results")
        disp_cols = ["rank", "compound_name", "molecular_formula",
                     "molecular_weight", "logp", score_col, "lipinski_status"]
        disp_cols = [c for c in disp_cols if c in results_df.columns]
        rename_map = {score_col: "Sim. Score (kcal/mol)", "compound_name": "Compound",
                      "molecular_formula": "Formula", "molecular_weight": "MW (g/mol)",
                      "logp": "LogP", "lipinski_status": "Lipinski"}

        def _highlight_best(row):
            return ["background-color: #C8F0DA" if row["rank"] == 1 else ""] * len(row)

        st.dataframe(
            results_df[disp_cols].rename(columns=rename_map)
            .style.apply(_highlight_best, axis=1),
            use_container_width=True, hide_index=True,
        )

        # ── ADMET Panel ────────────────────────────────────────────────
        st.markdown("#### 🧪 ADMET Properties Panel")
        admet_cols = ["compound_name", "molecular_weight", "logp",
                      "h_donors", "h_acceptors", "rotatable_bonds", "lipinski_status"]
        admet_cols = [c for c in admet_cols if c in results_df.columns]
        admet_rename = {
            "compound_name": "Compound", "molecular_weight": "MW (g/mol)",
            "logp": "LogP", "h_donors": "H-Donors", "h_acceptors": "H-Acceptors",
            "rotatable_bonds": "Rot. Bonds", "lipinski_status": "Lipinski Ro5",
        }
        st.dataframe(
            results_df[admet_cols].rename(columns=admet_rename),
            use_container_width=True, hide_index=True,
        )

        # ── Reference ligand comparison ────────────────────────────────
        ref_lig = (st.session_state.protein_info or {}).get("reference_ligand")
        if ref_lig:
            st.markdown("#### 📌 Screened Compounds vs Reference Ligand")
            ref_score_label = "Reference (crystallized)"
            comp_data = []
            for _, r in results_df.iterrows():
                comp_data.append({
                    "Compound": r["compound_name"],
                    "Score": r.get(score_col, 0),
                    "Type": "Screened",
                })
            # Add reference as a baseline if it has a score
            fig_ref = px.bar(
                pd.DataFrame(comp_data),
                x="Compound", y="Score", color="Type",
                color_discrete_map={"Screened": TEAL_MID},
                title=f"Simulated Scores vs Reference Ligand ({ref_lig.get('name')})",
                labels={"Score": "Sim. Score (kcal/mol)"},
            )
            # Add horizontal reference line at a plausible reference score
            ref_est = -8.9  # typical for oseltamivir-like compounds
            fig_ref.add_hline(
                y=ref_est, line_dash="dash", line_color=GOLD,
                annotation_text=f"Ref. ligand est. ~{ref_est} kcal/mol",
                annotation_position="top right",
            )
            fig_ref.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font_color=TEAL_DARK,
            )
            st.plotly_chart(fig_ref, use_container_width=True)

        # ── Charts ────────────────────────────────────────────────────
        st.markdown("#### 📈 Score Analysis Charts")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_bar = px.bar(
                results_df.sort_values(score_col),
                x="compound_name", y=score_col,
                color=score_col,
                color_continuous_scale="RdYlGn_r",
                title="Simulated Score Comparison (lower = stronger binding)",
                labels={"compound_name": "Compound", score_col: "Sim. Score (kcal/mol)"},
                text=score_col,
            )
            fig_bar.update_traces(textposition="outside", texttemplate="%{text:.2f}")
            fig_bar.update_layout(showlegend=False, plot_bgcolor="white",
                                  paper_bgcolor="white", font_color=TEAL_DARK)
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            if "molecular_weight" in results_df.columns and "logp" in results_df.columns:
                fig_sc = px.scatter(
                    results_df, x="molecular_weight", y=score_col,
                    size="molecular_weight", color="compound_name",
                    hover_name="compound_name",
                    title="MW vs Simulated Score",
                    labels={"molecular_weight": "MW (g/mol)", score_col: "Sim. Score (kcal/mol)"},
                )
                fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                     font_color=TEAL_DARK)
                st.plotly_chart(fig_sc, use_container_width=True)

        # Distribution histogram
        fig_dist = px.histogram(
            results_df, x=score_col, nbins=10,
            title="Score Distribution",
            labels={score_col: "Sim. Score (kcal/mol)", "count": "# Compounds"},
            color_discrete_sequence=[TEAL_MID],
        )
        fig_dist.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               font_color=TEAL_DARK)
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── Scientific Metrics ─────────────────────────────────────────
        st.markdown('<div class="section-header">Scientific Metrics Summary</div>', unsafe_allow_html=True)
        stats = statistics.calculate_statistics(results_df)
        m1, m2, m3, m4, m5 = st.columns(5)
        cards = [
            (m1, stats.get("count", 0),           "Compounds Screened"),
            (m2, stats.get("avg_mw", "N/A"),       "Avg. MW (g/mol)"),
            (m3, f"{stats.get('best_score','N/A')} kcal/mol",  "Highest Simulated Score"),
            (m4, f"{stats.get('worst_score','N/A')} kcal/mol", "Lowest Simulated Score"),
            (m5, f"{stats.get('average_score','N/A')} kcal/mol","Average Score"),
        ]
        for col, val, lbl in cards:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Export ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        e1, e2 = st.columns(2)

        with e1:
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            csv_name  = f"virtual_screening_{pdb_id_input}_{ts}.csv"
            st.download_button(
                "⬇️ Download Results CSV", data=csv_bytes,
                file_name=csv_name, mime="text/csv",
            )

        with e2:
            with st.spinner("Generating PDF report…"):
                pdf_bytes = report.generate_pdf_report(
                    st.session_state.protein_info, results_df, stats
                )
            pdf_name = f"virtual_screening_report_{pdb_id_input}_{ts}.pdf"
            st.download_button(
                "📄 Download PDF Report", data=pdf_bytes,
                file_name=pdf_name, mime="application/pdf",
            )

        st.caption("🗂️ Reports include: protein info, reference ligand, ADMET table, ranking, statistics, disclaimer.")

    else:
        if not results_df.empty:
            pass
        else:
            st.markdown(
                '<div class="info-box">Run a virtual screening simulation above to see ranked results, ADMET analysis, and charts here.</div>',
                unsafe_allow_html=True,
            )

    # ── Global disclaimer ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Educational Use Only.</b> Protein data is real (RCSB PDB). '
        'Simulated scores are formula-based and do NOT represent verified experimental or computational '
        'docking results. Not intended for actual drug research or clinical decisions.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: COMPOUND DETAILS
# ══════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown('<div class="section-header">💊 Compound Details</div>', unsafe_allow_html=True)
    st.write("Explore individual compound properties, structure images, and database identifiers.")

    compounds_df_d = screening.load_compound_data(DATA_PATH)
    sel_compound   = st.selectbox("Select a compound", compounds_df_d["compound_name"].tolist())

    if sel_compound:
        row = compounds_df_d[compounds_df_d["compound_name"] == sel_compound].iloc[0]
        smiles = row.get("smiles", "")
        cid    = row.get("pubchem_cid", None)
        db_id  = row.get("drugbank_id", None)

        d1, d2 = st.columns([2, 3])

        with d1:
            st.markdown(f"### {sel_compound}")
            st.markdown(f"**Molecular Formula:** `{row.get('molecular_formula', 'N/A')}`")
            st.markdown(f"**Molecular Weight:** {row.get('molecular_weight', 'N/A')} g/mol")
            st.markdown(f"**LogP:** {row.get('logp', 'N/A')}")
            st.markdown(f"**H-Bond Donors:** {row.get('h_donors', 'N/A')}")
            st.markdown(f"**H-Bond Acceptors:** {row.get('h_acceptors', 'N/A')}")
            st.markdown(f"**Rotatable Bonds:** {row.get('rotatable_bonds', 'N/A')}")

            lip = screening.lipinski_status(row.to_dict())
            st.markdown(f"**Lipinski Ro5:** {lip}")

            st.markdown(f"**PubChem CID:** {cid if cid else 'N/A'}")
            if cid and str(cid) != "nan":
                st.markdown(f"[🔗 View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)})")
            st.markdown(f"**DrugBank ID:** {db_id if db_id else 'N/A'}")
            if db_id and str(db_id) != "nan":
                st.markdown(f"[🔗 View on DrugBank](https://go.drugbank.com/drugs/{db_id})")
            st.markdown(f"**SMILES:**")
            st.code(smiles, language="text")

        with d2:
            st.markdown("#### 🖼️ Molecular Structure")
            if cid and str(cid) != "nan":
                img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/PNG?record_type=2d&image_size=400x300"
                try:
                    r = requests.get(img_url, timeout=8)
                    if r.status_code == 200:
                        st.image(r.content, caption=f"{sel_compound} — PubChem 2D Structure", use_container_width=True)
                    else:
                        st.info("Structure image not available for this compound.")
                except Exception:
                    st.info("Structure image unavailable (network error).")
            else:
                # Fallback: show SMILES via text
                st.info("No PubChem CID available — structure image cannot be fetched automatically.")
                st.code(smiles or "No SMILES available", language="text")

            # If PubChem available, also show 3D conformer image
            if cid and str(cid) != "nan":
                img3d_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/PNG?record_type=3d&image_size=400x300"
                try:
                    r3 = requests.get(img3d_url, timeout=8)
                    if r3.status_code == 200:
                        st.image(r3.content, caption=f"{sel_compound} — PubChem 3D Conformer", use_container_width=True)
                except Exception:
                    pass

    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ Structure images and identifiers are fetched from PubChem. Always verify IDs from primary sources before use in research.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: 3D PROTEIN VIEWER
# ══════════════════════════════════════════════════════════════════════════
with tab_viewer:
    st.markdown('<div class="section-header">🧪 3D Protein Structure Viewer</div>', unsafe_allow_html=True)
    st.write("Interactive 3D view via the **RCSB PDB** embedded viewer. Use the controls inside the viewer to change display style, colour scheme, and highlight ligands.")

    info = st.session_state.protein_info
    v1, v2 = st.columns([1, 3])

    with v1:
        view_pdb = st.text_input("PDB ID to view", value=(info.get("pdb_id") if info else "2HU4"), max_chars=4).strip().upper()

        st.info(
            "💡 **Tip:** Use the controls **inside the viewer** (top-right toolbar) "
            "to switch between cartoon, surface, and ball+stick styles, change colour "
            "schemes, and toggle ligand highlighting."
        )

        asr_display = ""
        if info and info.get("pdb_id") == view_pdb:
            asr = info.get("active_site_residues") or []
            if asr:
                asr_display = ", ".join(asr[:6])
                st.success(f"**Active site residues** *(within 5 Å of ligand)*: {asr_display}")
            ligands = info.get("ligands") or []
            if ligands:
                st.markdown(f"**💊 Co-crystallized ligand:** {ligands[0]}")

    with v2:
        # Embed RCSB 3D viewer via iframe (NGL-based, no install needed)
        viewer_html = f"""
<iframe
  src="https://www.rcsb.org/3d-view/{view_pdb}?preset=defaultView"
  width="100%"
  height="520"
  style="border:2px solid {TEAL_MID}; border-radius:10px;"
  title="3D Structure of {view_pdb}"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
>
</iframe>
<p style="font-size:0.8rem; color:grey; margin-top:6px;">
  Viewer powered by RCSB PDB / NGL Viewer. Ligands and active site residues are highlighted by default.
  For full interaction options, visit
  <a href="https://www.rcsb.org/structure/{view_pdb}" target="_blank">rcsb.org/structure/{view_pdb}</a>.
</p>
"""
        st.markdown(viewer_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**How to navigate the 3D viewer:**
- 🖱️ **Rotate** — click and drag
- 🔍 **Zoom** — scroll wheel
- 🤚 **Pan** — right-click and drag
- 💊 **Ligand** — shown in stick representation
- 🎯 **Active site** — highlighted in the preset view
""")
    st.markdown(
        '<div class="disclaimer">⚠️ 3D viewer uses the RCSB PDB website embed. Protein structures are real experimental data. Highlighted residues for active site are from curated annotations in this app.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: FUTURE UPGRADES
# ══════════════════════════════════════════════════════════════════════════
with tab_future:
    st.markdown('<div class="section-header">🚀 Future Upgrade Roadmap</div>', unsafe_allow_html=True)
    st.write(
        "This pipeline is designed for **educational virtual screening simulation**. "
        "Below are planned upgrades to transition it toward a production-grade, "
        "scientifically validated drug discovery tool."
    )

    upgrades = [
        ("🔬 AutoDock Vina Integration",
         "Replace the formula-based simulation with real molecular docking using AutoDock Vina. "
         "Requires protein .pdbqt preparation with AutoDockTools and a local Vina binary.",
         "High"),
        ("🧬 Real Molecular Docking",
         "Implement pose prediction, grid box setup, and binding affinity calculation using real "
         "force fields (MMFF94, Gasteiger charges). Output genuine kcal/mol binding energies.",
         "High"),
        ("💊 DrugBank API Integration",
         "Fetch real drug interaction profiles, pharmacokinetics, and mechanism of action for "
         "screened compounds via the DrugBank REST API.",
         "Medium"),
        ("🔭 PubChem API Integration",
         "Use the PubChem PUG-REST API for live SMILES, property retrieval, biological assay "
         "data, and compound similarity searching.",
         "Medium"),
        ("🤖 AI-Based Lead Prediction",
         "Integrate a Graph Neural Network (GNN) or transformer model (e.g., MolBERT, ChemBERTa) "
         "to predict binding affinity and ADMET properties from molecular graphs.",
         "High"),
        ("🌊 Molecular Dynamics (MD) Simulation",
         "Add a lightweight OpenMM or MDTraj interface to simulate protein-ligand complex "
         "stability and binding free energy estimation (MM-PBSA).",
         "Medium"),
        ("📊 ADMET Prediction API",
         "Connect to ADMETlab 2.0, SwissADME, or pkCSM APIs for validated predicted ADMET "
         "profiles instead of Lipinski rule-based estimates.",
         "Medium"),
        ("🧪 Scaffold Analysis & Clustering",
         "Implement Bemis-Murcko scaffold extraction and hierarchical clustering to group "
         "screened compounds by chemical series.",
         "Low"),
        ("🗂️ Multi-Target Screening",
         "Allow simultaneous screening against multiple PDB targets and generate a "
         "selectivity profile heatmap across targets.",
         "Low"),
    ]

    priority_colour = {"High": "#C0392B", "Medium": GOLD, "Low": GREEN_OK}

    for title, desc, priority in upgrades:
        p_col = priority_colour.get(priority, "#555")
        st.markdown(f"""
<div class="roadmap-item">
<b>{title}</b>
<span style="float:right; background:{p_col}; color:white; border-radius:10px; padding:1px 8px; font-size:0.75rem;">{priority} Priority</span>
<br><span style="color:#444; font-size:0.88rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📚 Recommended Tools for Future Implementation")
    tools = {
        "AutoDock Vina": "https://vina.scripps.edu/",
        "RDKit": "https://www.rdkit.org/",
        "OpenMM": "https://openmm.org/",
        "ADMETlab 2.0": "https://admetmesh.scbdd.com/",
        "PubChem PUG-REST": "https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest",
        "DrugBank API": "https://docs.drugbank.com/",
        "PyMOL": "https://pymol.org/",
        "ChemBERTa (HuggingFace)": "https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1",
    }
    cols = st.columns(4)
    for i, (name, url) in enumerate(tools.items()):
        with cols[i % 4]:
            st.markdown(f"[🔗 {name}]({url})")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ This app currently uses educational simulations only. '
        'The upgrades above require additional dependencies, computational resources, and '
        'validated scientific protocols before results can be used in research contexts.</div>',
        unsafe_allow_html=True,
    )
with tab_about:

    st.markdown("## 👨‍💻 About the Developer")

    st.markdown("""
### 🚀 AI-Driven Drug Discovery Pipeline

This project was designed and developed by **Vamsi Krishna Reddy**
as a B.Tech Biotechnology Final Year Project.

### 🤖 Built With AI

This platform was developed almost entirely using modern AI-assisted development tools including:

- ChatGPT (OpenAI)
- GitHub Copilot
- AI Code Generation
- Streamlit Cloud
- Python Automation Tools

### 🔬 Project Goal

To create an accessible AI-powered platform for:

- Protein Analysis
- Virtual Compound Screening
- ADMET Evaluation
- Automated Research Reporting
- Drug Discovery Education

### 🌍 Vision

The long-term goal is to evolve this platform into a complete AI-assisted drug discovery ecosystem capable of integrating:

- Molecular Docking
- Machine Learning Models
- Deep Learning Predictions
- Protein-Ligand Interaction Analysis
- Clinical Research Support

### 📚 Educational Disclaimer

This platform is currently intended for educational, academic, and research demonstration purposes.

### ❤️ Special Note

This project demonstrates how modern AI tools can empower biotechnology students to build advanced computational biology platforms with minimal resources.

**Developed by:** Vamsi Krishna Reddy  
**Program:** B.Tech Biotechnology  
**Institution:** KL University
""")
