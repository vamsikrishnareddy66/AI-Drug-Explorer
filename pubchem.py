"""
pubchem.py
----------
Live PubChem PUG-REST integration: compound property lookup and
bioassay summary fetching. ADMET rule-of-thumb scoring lives in
admet.py (kept separate so PubChem stays a pure data-fetch module).
"""

from typing import Optional

import requests
import streamlit as st

import admet

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_TIMEOUT = 8

COMPUTED_PROPERTIES = (
    "MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,"
    "HBondAcceptorCount,RotatableBondCount,TPSA,Complexity,"
    "HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,"
    "BondStereoCount,CovalentUnitCount"
)


def _resolve_cid(compound_name: str) -> Optional[int]:
    """Look up a PubChem CID by compound name."""
    search_url = (
        f"{PUBCHEM_BASE}/compound/name/"
        f"{requests.utils.quote(compound_name)}/cids/JSON"
    )
    r = requests.get(search_url, timeout=REQUEST_TIMEOUT)
    if r.status_code == 200:
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    return None


def _fetch_properties(cid: int) -> dict:
    """Fetch computed physicochemical properties for a CID."""
    prop_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{COMPUTED_PROPERTIES}/JSON"
    pr = requests.get(prop_url, timeout=REQUEST_TIMEOUT)
    if pr.status_code != 200:
        return {}
    props_data = pr.json().get("PropertyTable", {}).get("Properties", [{}])[0]
    return {
        "molecular_formula": props_data.get("MolecularFormula", "—"),
        "molecular_weight":  props_data.get("MolecularWeight", "—"),
        "xlogp":             props_data.get("XLogP", "—"),
        "h_donors":          props_data.get("HBondDonorCount", "—"),
        "h_acceptors":       props_data.get("HBondAcceptorCount", "—"),
        "rotatable_bonds":   props_data.get("RotatableBondCount", "—"),
        "tpsa":              props_data.get("TPSA", "—"),
        "complexity":        props_data.get("Complexity", "—"),
        "heavy_atoms":       props_data.get("HeavyAtomCount", "—"),
    }


def _fetch_bioassay_summary(cid: int) -> dict:
    """Fetch a count of active vs. total bioassays for a CID."""
    assay_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/assaysummary/JSON"
    ar = requests.get(assay_url, timeout=REQUEST_TIMEOUT)
    if ar.status_code != 200:
        return {}
    assays = ar.json().get("Table", {}).get("Row", [])
    active = sum(
        1 for row in assays
        if any(cell.get("Value", "") == "Active" for cell in (row.get("Cell") or []))
    )
    return {"bioassay_active_count": active, "bioassay_total_count": len(assays)}


def fetch_pubchem_admet(compound_name: str, cid: Optional[int] = None) -> dict:
    """
    Fetch live physicochemical + bioassay data from PubChem PUG-REST,
    plus a validated ADMET rule-flag summary (see admet.py).

    Results are cached in st.session_state.pubchem_admet_cache so repeat
    lookups within a session don't re-hit the API.
    """
    cache_key = f"pc_{cid or compound_name}"
    if cache_key in st.session_state.pubchem_admet_cache:
        return st.session_state.pubchem_admet_cache[cache_key]

    result = {"source": "pubchem_live", "compound": compound_name}

    try:
        if not cid:
            cid = _resolve_cid(compound_name)

        if cid:
            result["cid"] = cid
            result.update(_fetch_properties(cid))
            result.update(_fetch_bioassay_summary(cid))
            result["admet_flags"] = admet.compute_admet_flags(result)

    except Exception as e:
        result["error"] = str(e)

    st.session_state.pubchem_admet_cache[cache_key] = result
    return result
