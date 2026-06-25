"""
config.py
---------
Single source of truth for all app-wide constants, settings, and feature
flags. Nothing in here should ever be hardcoded back into app.py — import
from this module instead.
"""

# ── App metadata ─────────────────────────────────────────────────────────
APP_VERSION = "5.0"
APP_TITLE   = "Drug Discovery Pipeline"
DEVELOPER   = "N. Vamsi Krishna Reddy"
INSTITUTION = "KL University"

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH   = "compounds.csv"
EXPORTS_DIR = "exports"

# ── Scoring ───────────────────────────────────────────────────────────────
SCORE_COL = "simulated_score"

# ── Colour palette (used by theme.py and any module rendering Plotly/HTML) ─
TEAL_DARK  = "#071A2F"
TEAL_MID   = "#1E3A8A"
TEAL_LIGHT = "#00E5FF"
GOLD       = "#FFD166"
GREEN_OK   = "#00F5A0"
BG_CARD    = "#16213E"
PINK       = "#FF2E88"

# ── External API base URLs ───────────────────────────────────────────────
RCSB_ENTRY_API   = "https://data.rcsb.org/rest/v1/core/entry/{}"
RCSB_POLYMER_API = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"
RCSB_NONPOLY_API = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{}/{}"
RCSB_FILE_DL     = "https://files.rcsb.org/download/{}.pdb"

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
GEMINI_MODEL     = "gemini-2.5-flash"
GEMINI_API_URL   = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

# ── Request settings ──────────────────────────────────────────────────────
REQUEST_TIMEOUT     = 10   # seconds, general API calls
AI_REQUEST_TIMEOUT  = 45   # seconds, Gemini calls (can be slower)
FILE_DL_TIMEOUT     = 15   # seconds, structure file downloads

# ── Feature flags ─────────────────────────────────────────────────────────
FEATURES = {
    "pubchem_live_admet": True,
    "drugbank_tab":       True,
    "scaffold_clustering": True,
    "multi_target_screening": True,
    "autodock_vina":      True,
    "ai_assistant":       True,
}

# ── Session-state defaults (imported by app.py at startup) ──────────────
SESSION_DEFAULTS = {
    "history":              [],
    "results_df":            None,   # set to pd.DataFrame() at runtime in app.py
    "protein_info":          None,
    "reports_count":         0,
    "ai_chat":               [],
    "multi_target_results":  {},
    "pubchem_admet_cache":   {},
    "drugbank_cache":        {},
    "scaffold_cache":        {},
    "vina_available":        None,
    "pdf_ready":             False,
    "pdf_bytes":             None,
    "pdf_pdb_id":            None,
}
