"""
admet.py
--------
Rule-of-thumb ADMET (Absorption, Distribution, Metabolism, Excretion,
Toxicity) scoring, computed from physicochemical properties returned
by pubchem.py. Pure functions — no Streamlit/session-state dependency,
so this module is independently testable.

Implements:
    - Lipinski Rule of Five
    - Veber rule (oral absorption proxy)
    - Crude BBB penetration heuristic
    - Crude hERG liability heuristic
    - An aggregate drug-likeness score (0–5)
"""

from typing import Optional


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compute_admet_flags(props: dict) -> dict:
    """
    Compute validated ADMET flags from physicochemical data.

    Parameters
    ----------
    props : dict
        Expects keys: molecular_weight, xlogp, h_donors, h_acceptors,
        rotatable_bonds, tpsa (as returned by pubchem.fetch_pubchem_admet).

    Returns
    -------
    dict
        Mapping of rule name -> (status, note), where status is one of
        "PASS", "WARN", "FAIL".
    """
    flags = {}

    mw   = _safe_float(props.get("molecular_weight"))
    logp = _safe_float(props.get("xlogp"))
    hbd  = _safe_float(props.get("h_donors"))
    hba  = _safe_float(props.get("h_acceptors"))
    rb   = _safe_float(props.get("rotatable_bonds"))
    tpsa = _safe_float(props.get("tpsa"))

    # ── Lipinski Rule of Five ───────────────────────────────────────
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

    # ── Veber rule (oral absorption proxy): TPSA <= 140, RotBonds <= 10 ─
    if tpsa is not None and rb is not None:
        if tpsa <= 140 and rb <= 10:
            flags["Oral Absorption"] = ("PASS", f"TPSA={tpsa}, RotBonds={rb}")
        else:
            flags["Oral Absorption"] = ("WARN", f"TPSA={tpsa}, RotBonds={rb}")

    # ── BBB penetration (crude heuristic) ───────────────────────────
    # MW < 450, LogP between 1-3, TPSA < 90
    if mw is not None and logp is not None:
        bbb_ok = mw < 450 and 1 <= logp <= 3 and (tpsa is None or tpsa < 90)
        flags["BBB Penetration"] = (
            "PASS" if bbb_ok else "WARN",
            f"MW={mw}, LogP={logp}"
        )

    # ── hERG liability (crude heuristic): LogP > 3.7 and MW > 300 ───
    if logp is not None and mw is not None:
        herg_risk = logp > 3.7 and mw > 300
        flags["hERG Liability"] = (
            "WARN" if herg_risk else "PASS",
            "High lipophilicity risk" if herg_risk else "Low risk"
        )

    # ── Aggregate drug-likeness score (0-5) ─────────────────────────
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
