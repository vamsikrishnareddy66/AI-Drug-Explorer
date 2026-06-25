"""
drugbank.py
-----------
DrugBank data fetching: live API (requires DRUGBANK_API_KEY) with a
curated offline fallback for common drugs used in this project's
virtual screening dataset. UI rendering stays in app.py — this module
only fetches and caches data.
"""

import os
from typing import Optional

import requests
import streamlit as st

DRUGBANK_API_BASE = "https://api.drugbank.com/v1/drugs"
REQUEST_TIMEOUT = 10

# ── Curated offline data for common drugs ──────────────────────────────
# Used as a fallback when no DRUGBANK_API_KEY is configured, or when the
# compound has no associated drugbank_id in compounds.csv.
CURATED_FALLBACK = {
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


def get_drugbank_key() -> Optional[str]:
    """Resolve the DrugBank API key: st.secrets first, then environment variable."""
    try:
        return st.secrets["DRUGBANK_API_KEY"]
    except Exception:
        return os.environ.get("DRUGBANK_API_KEY")


def _curated_fallback(compound_name: str) -> dict:
    """Look up curated offline data for well-known drugs by substring match."""
    name_lower = compound_name.lower()
    for key, data in CURATED_FALLBACK.items():
        if key in name_lower:
            return {**data, "compound": compound_name}
    return {"source": "not_found", "note": "No curated data. Add DRUGBANK_API_KEY for live data."}


def _fetch_live(drugbank_id: str, api_key: str) -> dict:
    """Fetch a single drug record from the live DrugBank API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.get(f"{DRUGBANK_API_BASE}/{drugbank_id}", headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return {}

    data = r.json()
    return {
        "source":           "drugbank_live",
        "name":             data.get("name", "—"),
        "description":      data.get("description", "—"),
        "mechanism":        data.get("mechanism_of_action", "—"),
        "indication":       data.get("indication", "—"),
        "pharmacodynamics": data.get("pharmacodynamics", "—"),
        "half_life":        data.get("half_life", "—"),
        "protein_binding":  data.get("protein_binding", "—"),
        "metabolism":       data.get("metabolism", "—"),
        "toxicity":         data.get("toxicity", "—"),
        "categories":       [c.get("name", "") for c in data.get("categories", [])],
        "interactions": [
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
    }


def fetch_drugbank_info(compound_name: str, drugbank_id: Optional[str] = None) -> dict:
    """
    Fetch drug information from DrugBank's live API.
    Falls back to curated offline data when no API key is set, the
    compound has no drugbank_id, or the live call fails.

    Results are cached in st.session_state.drugbank_cache per session.
    """
    cache_key = f"db_{drugbank_id or compound_name}"
    if cache_key in st.session_state.drugbank_cache:
        return st.session_state.drugbank_cache[cache_key]

    result = {
        "compound": compound_name,
        "drugbank_id": drugbank_id,
        "source": "unavailable",
    }

    api_key = get_drugbank_key()
    if api_key and drugbank_id:
        try:
            live = _fetch_live(drugbank_id, api_key)
            if live:
                result.update(live)
        except Exception as e:
            result["error"] = str(e)

    if result["source"] == "unavailable":
        result.update(_curated_fallback(compound_name))

    st.session_state.drugbank_cache[cache_key] = result
    return result
