"""
report.py
---------
Generates a comprehensive PDF report for the Virtual Screening Simulation.

Includes:
 - Project title & date
 - Protein information (with chains, ligands, active site)
 - Reference ligand section
 - ADMET / compound properties table
 - Ranked results table
 - Summary statistics
 - Disclaimer
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

# ── Colour palette ────────────────────────────────────────────────────────
TEAL_DARK  = colors.HexColor("#0D4C6E")
TEAL_MID   = colors.HexColor("#1A7FA3")
TEAL_LIGHT = colors.HexColor("#D6EEF8")
GOLD       = colors.HexColor("#B5892A")
GREY_LIGHT = colors.HexColor("#F5F5F5")


def _make_styles():
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle(
        "TitleS", parent=styles["Title"], fontSize=22,
        textColor=TEAL_DARK, spaceAfter=4,
    )
    sub_s = ParagraphStyle(
        "SubS", parent=styles["Normal"], fontSize=11,
        textColor=colors.grey, spaceAfter=6,
    )
    h2_s = ParagraphStyle(
        "H2S", parent=styles["Heading2"], textColor=TEAL_DARK,
        fontSize=13, spaceBefore=10, spaceAfter=4,
    )
    h3_s = ParagraphStyle(
        "H3S", parent=styles["Heading3"], textColor=TEAL_MID,
        fontSize=11, spaceBefore=6, spaceAfter=2,
    )
    normal_s  = styles["Normal"]
    small_s   = ParagraphStyle("SmallS", parent=normal_s, fontSize=8, textColor=colors.grey)
    bold_s    = ParagraphStyle("BoldS",  parent=normal_s, fontName="Helvetica-Bold")
    warn_s    = ParagraphStyle("WarnS",  parent=small_s,  textColor=GOLD, fontSize=9)
    return title_s, sub_s, h2_s, h3_s, normal_s, small_s, bold_s, warn_s


def _kv(key, value, style) -> list:
    return [Paragraph(f"<b>{key}:</b> {value or 'N/A'}", style), Spacer(1, 0.1 * cm)]


def generate_pdf_report(protein_info: dict, results_df, stats: dict) -> bytes:
    """Build and return the PDF as bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )

    ts_s, sub_s, h2_s, h3_s, norm_s, small_s, bold_s, warn_s = _make_styles()
    elems = []

    # ── Cover / Title ─────────────────────────────────────────────────────
    elems.append(Paragraph("Virtual Screening Simulation", ts_s))
    elems.append(Paragraph("Drug Discovery Pipeline — B.Tech Biotechnology Final Year Project", sub_s))
    elems.append(Paragraph(f"Report generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}", small_s))
    elems.append(Spacer(1, 0.3 * cm))
    elems.append(HRFlowable(width="100%", color=TEAL_DARK, thickness=2))
    elems.append(Spacer(1, 0.4 * cm))

    # ── Protein Information ───────────────────────────────────────────────
    elems.append(Paragraph("1. Protein Target Information", h2_s))
    if protein_info:
        fields = [
            ("PDB ID",              protein_info.get("pdb_id")),
            ("Protein Name",        protein_info.get("name")),
            ("Organism",            protein_info.get("organism")),
            ("Experimental Method", protein_info.get("method")),
            ("Resolution",          f"{protein_info.get('resolution')} Å" if protein_info.get("resolution") else "N/A"),
            ("Chain(s)",            ", ".join(protein_info.get("chains", [])) or "See PDB entry"),
            ("Ligand(s) in Structure", ", ".join(protein_info.get("ligands", [])) or "None detected"),
            ("Active Site Residues", ", ".join(protein_info.get("active_site_residues", [])) or "Not annotated"),
        ]
        for k, v in fields:
            elems += _kv(k, v, norm_s)

    elems.append(Spacer(1, 0.5 * cm))

    # ── Reference Ligand ─────────────────────────────────────────────────
    ref_lig = protein_info.get("reference_ligand") if protein_info else None
    if ref_lig:
        elems.append(Paragraph("2. Reference Ligand (Co-crystallized)", h2_s))
        elems += _kv("Name",            ref_lig.get("name"),    norm_s)
        elems += _kv("Formula",         ref_lig.get("formula"),  norm_s)
        elems += _kv("Molecular Weight", f"{ref_lig.get('mw')} g/mol", norm_s)
        elems += _kv("PubChem CID",     ref_lig.get("pubchem_cid"), norm_s)
        elems += _kv("SMILES",          ref_lig.get("smiles"),   norm_s)
        elems.append(Paragraph(f"<i>{ref_lig.get('notes', '')}</i>", small_s))
        elems.append(Spacer(1, 0.5 * cm))
        sec_num = 3
    else:
        sec_num = 2

    # ── ADMET / Compound Properties Table ────────────────────────────────
    elems.append(Paragraph(f"{sec_num}. ADMET Properties (Lipinski Analysis)", h2_s))
    elems.append(Paragraph(
        "ADMET properties are calculated from compound structure. "
        "Lipinski Rule of Five: MW ≤ 500 g/mol, LogP ≤ 5, H-donors ≤ 5, H-acceptors ≤ 10. "
        "Violations suggest poor oral bioavailability.",
        small_s,
    ))
    elems.append(Spacer(1, 0.2 * cm))
    sec_num += 1

    if not results_df.empty:
        admet_header = [
            "Compound", "MW\n(g/mol)", "LogP", "H-Don", "H-Acc", "RotBonds", "Lipinski"
        ]
        admet_data = [admet_header]
        for _, row in results_df.iterrows():
            admet_data.append([
                row.get("compound_name", ""),
                f"{row.get('molecular_weight', 0):.2f}",
                f"{row.get('logp', 0):.2f}",
                str(row.get("h_donors", "N/A")),
                str(row.get("h_acceptors", "N/A")),
                str(row.get("rotatable_bonds", "N/A")),
                str(row.get("lipinski_status", "N/A")).replace("✅", "Pass").replace("❌", "Fail").replace("⚠️", "Borderline"),
            ])
        admet_table = Table(
            admet_data, hAlign="LEFT",
            colWidths=[3.8*cm, 1.8*cm, 1.5*cm, 1.4*cm, 1.4*cm, 1.8*cm, 3.5*cm]
        )
        admet_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), TEAL_MID),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_LIGHT]),
            ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ]))
        elems.append(admet_table)
        elems.append(Spacer(1, 0.5 * cm))

    # ── Ranked Results Table ──────────────────────────────────────────────
    elems.append(Paragraph(f"{sec_num}. Virtual Screening Ranking", h2_s))
    elems.append(Paragraph(
        "Compounds ranked by simulated binding score. Lower (more negative) = stronger simulated binding.",
        small_s,
    ))
    elems.append(Spacer(1, 0.2 * cm))
    sec_num += 1

    score_col = "simulated_score" if "simulated_score" in results_df.columns else "docking_score"
    if not results_df.empty:
        rank_header = ["Rank", "Compound", "Formula", "MW (g/mol)", f"Sim. Score (kcal/mol)"]
        rank_data = [rank_header]
        for _, row in results_df.iterrows():
            rank_data.append([
                str(row.get("rank", "")),
                row.get("compound_name", ""),
                row.get("molecular_formula", ""),
                f"{row.get('molecular_weight', 0):.2f}",
                f"{row.get(score_col, 0):.2f}",
            ])
        rank_table = Table(rank_data, hAlign="LEFT", colWidths=[1.5*cm, 4*cm, 3.2*cm, 2.8*cm, 4*cm])
        rank_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), TEAL_DARK),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_LIGHT]),
            ("BACKGROUND",   (0, 1), (-1, 1), colors.HexColor("#C8E6C9")),  # best row
        ]))
        elems.append(rank_table)
        elems.append(Spacer(1, 0.5 * cm))

    # ── Summary Statistics ────────────────────────────────────────────────
    elems.append(Paragraph(f"{sec_num}. Summary Statistics", h2_s))
    sec_num += 1
    stats_data = [
        ["Metric", "Value"],
        ["Compounds Screened",            str(stats.get("count", 0))],
        ["Average Molecular Weight",      f"{stats.get('avg_mw', 'N/A')} g/mol"],
        ["Highest Simulated Score (Best)",f"{stats.get('best_score', 'N/A')} kcal/mol"],
        ["Lowest Simulated Score (Worst)",f"{stats.get('worst_score', 'N/A')} kcal/mol"],
        ["Average Simulated Score",       f"{stats.get('average_score', 'N/A')} kcal/mol"],
    ]
    stats_table = Table(stats_data, hAlign="LEFT", colWidths=[7*cm, 7*cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), TEAL_MID),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TEAL_LIGHT]),
    ]))
    elems.append(stats_table)
    elems.append(Spacer(1, 0.5 * cm))

    # ── Future Upgrades ───────────────────────────────────────────────────
    elems.append(Paragraph(f"{sec_num}. Future Upgrade Roadmap", h2_s))
    upgrades = [
        "• AutoDock Vina integration for real molecular docking",
        "• DrugBank API integration for real drug interaction data",
        "• PubChem API integration for live compound property lookup",
        "• AI-based lead prediction using graph neural networks",
        "• Molecular dynamics (MD) simulation with OpenMM",
        "• ADMET prediction using ML models (e.g., ADMETlab 2.0 API)",
    ]
    for u in upgrades:
        elems.append(Paragraph(u, norm_s))
    elems.append(Spacer(1, 0.6 * cm))

    # ── Disclaimer ────────────────────────────────────────────────────────
    elems.append(HRFlowable(width="100%", color=TEAL_DARK, thickness=1))
    elems.append(Spacer(1, 0.3 * cm))
    elems.append(Paragraph("⚠️  EDUCATIONAL DISCLAIMER", bold_s))
    elems.append(Paragraph(
        "Scores shown are EDUCATIONAL SIMULATIONS and not actual docking results. "
        "Protein structures are retrieved live from the RCSB Protein Data Bank (real data). "
        "Simulated scores are derived from a formula using compound physicochemical properties "
        "(LogP, MW) and protein metadata (resolution, entity count) — they do NOT represent "
        "verified computational or experimental binding energies and MUST NOT be used for "
        "actual drug research, clinical decisions, or publication without validated docking software.",
        warn_s,
    ))

    doc.build(elems)
    buffer.seek(0)
    return buffer.getvalue()
