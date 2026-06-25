"""
admet.py
--------
Project: AI Drug Discovery Platform
Description: Central ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 
             prediction engine. Computes rule-of-thumb heuristic scores from 
             physicochemical properties.
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

__version__ = "2.1.0"
__all__ = [
    "compute_complete_admet",
    "compute_admet_flags",
    "DescriptorValidator",
    "RuleResult",
    "PredictionResult",
    "InteractionPrediction",
    "ADMETSummary",
    "StructuredRecommendation",
    "Status",
    "PredictionLevel"
]

# ── Logging Setup ────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────
class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

class PredictionLevel(str, Enum):
    EXCELLENT = "EXCELLENT"
    VERY_GOOD = "VERY GOOD"
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    LOW = "LOW"
    HIGH = "HIGH"
    POOR = "POOR"
    UNSUITABLE = "UNSUITABLE"
    LIKELY = "LIKELY"
    UNLIKELY = "UNLIKELY"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNKNOWN = "UNKNOWN"

# ── Scientific Constants & Penalties ─────────────────────────────────────
LIPINSKI_PENALTY = 1.0 / 4.0
VEBER_PENALTY = 1.0 / 2.0
GHOSE_PENALTY = 1.0 / 4.0
EGAN_PENALTY = 1.0 / 2.0
MUEGGE_PENALTY = 1.0 / 7.0
GSK_PENALTY = 1.0 / 2.0

SCORE_WEIGHT_DRUGLIKENESS = 0.30
SCORE_WEIGHT_ABSORPTION = 0.25
SCORE_WEIGHT_DISTRIBUTION = 0.15
SCORE_WEIGHT_METABOLISM = 0.10
SCORE_WEIGHT_TOXICITY = 0.20

LIPINSKI_MW_MAX, LIPINSKI_LOGP_MAX, LIPINSKI_HBD_MAX, LIPINSKI_HBA_MAX = 500.0, 5.0, 5, 10
VEBER_TPSA_MAX, VEBER_ROTATABLE_MAX = 140.0, 10
GHOSE_MW_MIN, GHOSE_MW_MAX = 160.0, 480.0
GHOSE_LOGP_MIN, GHOSE_LOGP_MAX = -0.4, 5.6
GHOSE_ATOM_MIN, GHOSE_ATOM_MAX = 20, 70
GHOSE_MR_MIN, GHOSE_MR_MAX = 40.0, 130.0
EGAN_LOGP_MAX, EGAN_TPSA_MAX = 5.88, 131.6
MUEGGE_MW_MIN, MUEGGE_MW_MAX = 200.0, 600.0
MUEGGE_LOGP_MIN, MUEGGE_LOGP_MAX = -2.0, 5.0
MUEGGE_TPSA_MAX, MUEGGE_RINGS_MIN = 150.0, 1
MUEGGE_HBD_MAX, MUEGGE_HBA_MAX, MUEGGE_RB_MAX = 5, 10, 15
PFIZER_LOGP_MIN, PFIZER_TPSA_MAX = 3.0, 75.0
GSK_LOGP_MAX, GSK_MW_MAX = 4.0, 400.0
BBB_MW_MAX, BBB_LOGP_MIN, BBB_LOGP_MAX, BBB_TPSA_MAX = 400.0, 1.5, 2.7, 90.0

# ── Dataclasses ──────────────────────────────────────────────────────────
@dataclass
class RuleResult:
    name: str
    status: Status
    score: float
    reason: str
    violations: int
    color: str

@dataclass
class PredictionResult:
    endpoint: str
    prediction: str
    score: float
    confidence: float
    reason: str
    color: str

@dataclass
class Metadata:
    version: str
    timestamp: str
    compound_name: Optional[str] = None
    pubchem_cid: Optional[str] = None
    pdb_id: Optional[str] = None

@dataclass
class InteractionPrediction:
    target_name: str
    interaction_type: str
    prediction: str
    confidence: float
    reason: str = field(default="")
    color: str = field(default="yellow")

@dataclass
class StructuredRecommendation:
    strengths: List[str]
    limitations: List[str]
    overall_recommendation: str

@dataclass
class ADMETSummary:
    metadata: Metadata
    drug_likeness: Dict[str, RuleResult]
    absorption: Dict[str, PredictionResult]
    distribution: Dict[str, PredictionResult]
    metabolism: Dict[str, PredictionResult]
    excretion: Dict[str, PredictionResult]
    toxicity: Dict[str, PredictionResult]
    drug_likeness_score: float
    admet_score: float
    confidence_score: float
    classification: str
    recommendation: StructuredRecommendation

    def as_dict(self) -> dict:
        return asdict(self)

# ── Utility & Validation Functions ───────────────────────────────────────
def safe_float(val: Any) -> Optional[float]:
    try: return float(val) if val is not None else None
    except (TypeError, ValueError): return None

def safe_int(val: Any) -> Optional[int]:
    try: return int(float(val)) if val is not None else None
    except (TypeError, ValueError): return None

def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(val, max_val))

def status_color(status: str) -> str:
    s = status.upper()
    if s in ("PASS", "HIGH", "EXCELLENT", "VERY GOOD", "GOOD", "LIKELY", "HIGHLY SOLUBLE"):
        return "green"
    elif s in ("WARN", "MODERATE", "MEDIUM", "MODERATELY SOLUBLE", "INCONCLUSIVE"):
        return "yellow"
    else:
        return "red"

def tox_color(risk_level: str) -> str:
    r = risk_level.upper()
    if r in ("LOW", "UNLIKELY", "PASS"): return "green"
    elif r in ("MODERATE", "WARN", "INCONCLUSIVE"): return "yellow"
    else: return "red"

class DescriptorValidator:
    def __init__(self, props: dict):
        self.mw = safe_float(props.get("molecular_weight"))
        self.logp = safe_float(props.get("xlogp"))
        self.hbd = safe_float(props.get("h_donors"))
        self.hba = safe_float(props.get("h_acceptors"))
        self.rb = safe_float(props.get("rotatable_bonds"))
        self.tpsa = safe_float(props.get("tpsa"))
        self.exact_mass = safe_float(props.get("exact_mass", self.mw))
        self.heavy_atoms = safe_int(props.get("heavy_atom_count", props.get("non_hydrogen_atoms")))
        self.rings = safe_int(props.get("ring_count"))
        self.mr = safe_float(props.get("molar_refractivity"))

        self.required = [self.mw, self.logp, self.hbd, self.hba, self.rb, self.tpsa]
        self.missing_count = sum([1 for x in self.required if x is None])
        self.completeness = (len(self.required) - self.missing_count) / len(self.required) if self.required else 0.0

# ── Drug-Likeness Rules ──────────────────────────────────────────────────
def eval_lipinski(v: DescriptorValidator) -> RuleResult:
    violations = 0
    if v.mw is not None and v.mw > LIPINSKI_MW_MAX: violations += 1
    if v.logp is not None and v.logp > LIPINSKI_LOGP_MAX: violations += 1
    if v.hbd is not None and v.hbd > LIPINSKI_HBD_MAX: violations += 1
    if v.hba is not None and v.hba > LIPINSKI_HBA_MAX: violations += 1
    status = Status.PASS if violations == 0 else (Status.WARN if violations == 1 else Status.FAIL)
    score = max(0.0, 1.0 - (violations * LIPINSKI_PENALTY))
    return RuleResult("Lipinski Ro5", status, score, f"{violations} violation(s)", violations, status_color(status))

def eval_veber(v: DescriptorValidator) -> RuleResult:
    violations = 0
    if v.tpsa is not None and v.tpsa > VEBER_TPSA_MAX: violations += 1
    if v.rb is not None and v.rb > VEBER_ROTATABLE_MAX: violations += 1
    status = Status.PASS if violations == 0 else (Status.WARN if violations == 1 else Status.FAIL)
    score = max(0.0, 1.0 - (violations * VEBER_PENALTY))
    return RuleResult("Veber Rule", status, score, f"TPSA={v.tpsa or '?'}, RotBonds={v.rb or '?'}", violations, status_color(status))

def eval_ghose(v: DescriptorValidator) -> RuleResult:
    violations = 0
    if v.mw is not None and not (GHOSE_MW_MIN <= v.mw <= GHOSE_MW_MAX): violations += 1
    if v.logp is not None and not (GHOSE_LOGP_MIN <= v.logp <= GHOSE_LOGP_MAX): violations += 1
    if v.heavy_atoms is not None and not (GHOSE_ATOM_MIN <= v.heavy_atoms <= GHOSE_ATOM_MAX): violations += 1
    if v.mr is not None and not (GHOSE_MR_MIN <= v.mr <= GHOSE_MR_MAX): violations += 1
    status = Status.PASS if violations == 0 else (Status.WARN if violations <= 2 else Status.FAIL)
    score = clamp(1.0 - (violations * GHOSE_PENALTY))
    return RuleResult("Ghose Filter", status, score, f"{violations} violation(s)", violations, status_color(status))

def eval_egan(v: DescriptorValidator) -> RuleResult:
    violations = 0
    if v.logp is not None and v.logp > EGAN_LOGP_MAX: violations += 1
    if v.tpsa is not None and v.tpsa > EGAN_TPSA_MAX: violations += 1
    status = Status.PASS if violations == 0 else (Status.WARN if violations == 1 else Status.FAIL)
    score = max(0.0, 1.0 - (violations * EGAN_PENALTY))
    return RuleResult("Egan Rule", status, score, f"LogP <= {EGAN_LOGP_MAX}, TPSA <= {EGAN_TPSA_MAX}", violations, status_color(status))

def eval_muegge(v: DescriptorValidator) -> RuleResult:
    violations = 0
    if v.mw is not None and not (MUEGGE_MW_MIN <= v.mw <= MUEGGE_MW_MAX): violations += 1
    if v.logp is not None and not (MUEGGE_LOGP_MIN <= v.logp <= MUEGGE_LOGP_MAX): violations += 1
    if v.tpsa is not None and v.tpsa > MUEGGE_TPSA_MAX: violations += 1
    if v.rings is not None and v.rings < MUEGGE_RINGS_MIN: violations += 1
    if v.hbd is not None and v.hbd > MUEGGE_HBD_MAX: violations += 1
    if v.hba is not None and v.hba > MUEGGE_HBA_MAX: violations += 1
    if v.rb is not None and v.rb > MUEGGE_RB_MAX: violations += 1
    status = Status.PASS if violations == 0 else (Status.WARN if violations <= 2 else Status.FAIL)
    score = clamp(1.0 - (violations * MUEGGE_PENALTY))
    return RuleResult("Muegge Rule", status, score, f"{violations} violation(s)", violations, status_color(status))

def eval_pfizer(v: DescriptorValidator) -> RuleResult:
    risk = (v.logp is not None and v.tpsa is not None and v.logp > PFIZER_LOGP_MIN and v.tpsa < PFIZER_TPSA_MAX)
    status = Status.WARN if risk else Status.PASS
    return RuleResult("Pfizer 3/75", status, 0.0 if risk else 1.0, "Toxicity risk" if risk else "Favorable", 1 if risk else 0, tox_color("HIGH" if risk else "LOW"))

def eval_gsk(v: DescriptorValidator) -> RuleResult:
    favorable = not ((v.mw is not None and v.mw > GSK_MW_MAX) or (v.logp is not None and v.logp > GSK_LOGP_MAX))
    status = Status.PASS if favorable else Status.WARN
    score = 1.0 if favorable else max(0.0, 1.0 - GSK_PENALTY)
    return RuleResult("GSK 4/400", status, score, "Favorable" if favorable else "Suboptimal", 0 if favorable else 1, status_color(status))

# ── Prediction Logic ─────────────────────────────────────────────────────
def predict_gi_absorption(v: DescriptorValidator) -> PredictionResult:
    if v.tpsa is None: return PredictionResult("GI Absorption", PredictionLevel.UNKNOWN, 0.0, 0.0, "Missing TPSA", "red")
    score = 1.0 if (v.tpsa <= VEBER_TPSA_MAX and (v.rb is None or v.rb <= VEBER_ROTATABLE_MAX)) else 0.3
    level = PredictionLevel.HIGH if score == 1.0 else PredictionLevel.LOW
    return PredictionResult("GI Absorption", level, score, v.completeness, "Favorable TPSA & RotBonds" if score==1.0 else "Poor TPSA or RotBonds", status_color(level))

def predict_solubility(v: DescriptorValidator) -> PredictionResult:
    if None in [v.logp, v.mw, v.tpsa]: return PredictionResult("Water Solubility", PredictionLevel.UNKNOWN, 0.0, 0.0, "Missing Descriptors", "red")
    score = 1.0
    if v.logp > 4: score -= 0.4
    if v.mw > 400: score -= 0.3
    if v.tpsa < 20 or v.tpsa > 120: score -= 0.3
    score = clamp(score)
    level = "Highly Soluble" if score > 0.7 else ("Moderately Soluble" if score > 0.4 else "Poorly Soluble")
    return PredictionResult("Water Solubility", level, score, v.completeness, f"MW/LogP/TPSA based", status_color(level))

def predict_bioavailability(v: DescriptorValidator, dl_rules: dict) -> PredictionResult:
    scores = [dl_rules["Lipinski"].score, dl_rules["Veber"].score, dl_rules["Ghose"].score, dl_rules["Egan"].score]
    avg_score = sum(scores) / len(scores)
    level = PredictionLevel.HIGH if avg_score >= 0.75 else (PredictionLevel.MODERATE if avg_score >= 0.5 else PredictionLevel.LOW)
    return PredictionResult("Bioavailability", level, avg_score, v.completeness, f"Derived from 4 rules", status_color(level))

def predict_pgp(v: DescriptorValidator) -> PredictionResult:
    risk_factors = 0
    if v.mw is not None and v.mw > 400: risk_factors += 1
    if v.hba is not None and v.hba > 4: risk_factors += 1
    if v.tpsa is not None and v.tpsa > 60: risk_factors += 1
    if v.logp is not None and v.logp > 2: risk_factors += 1
    if v.rb is not None and v.rb > 4: risk_factors += 1
    score = risk_factors / 5.0
    is_substrate = score >= 0.6
    level = PredictionLevel.LIKELY if is_substrate else PredictionLevel.UNLIKELY
    return PredictionResult("P-gp Substrate", level, score, v.completeness, "Multiparameter heuristic", tox_color("MODERATE" if is_substrate else "LOW"))

def predict_bbb(v: DescriptorValidator) -> PredictionResult:
    if v.mw is None or v.logp is None:
        return PredictionResult("BBB Penetration", PredictionLevel.UNKNOWN, 0.0, 0.0, "Missing Descriptors", "red")
    score = 1.0
    if v.mw > BBB_MW_MAX: score -= 0.4
    if not (BBB_LOGP_MIN <= v.logp <= BBB_LOGP_MAX): score -= 0.4
    if v.tpsa is not None and v.tpsa > BBB_TPSA_MAX: score -= 0.3
    score = clamp(score)
    level = PredictionLevel.HIGH if score > 0.8 else (PredictionLevel.MODERATE if score > 0.4 else PredictionLevel.LOW)
    return PredictionResult("BBB Penetration", level, score, v.completeness, f"BBB Score: {int(score*100)}/100", status_color(level))

def predict_ppb(v: DescriptorValidator) -> PredictionResult:
    if v.logp is None: return PredictionResult("Plasma Protein Binding", PredictionLevel.UNKNOWN, 0.0, 0.0, "Missing LogP", "red")
    score = clamp(v.logp / 5.0)
    level = PredictionLevel.HIGH if v.logp > 4 else (PredictionLevel.MODERATE if v.logp > 2 else PredictionLevel.LOW)
    return PredictionResult("Plasma Protein Binding", level, score, v.completeness, "Driven by lipophilicity", status_color(level))

def _cyp_heuristic(v: DescriptorValidator, mw_thresh: float, logp_thresh: float, name: str) -> PredictionResult:
    is_risk = (v.mw is not None and v.mw > mw_thresh) and (v.logp is not None and v.logp > logp_thresh)
    score = 1.0 if is_risk else 0.0
    risk = PredictionLevel.HIGH if is_risk else PredictionLevel.LOW
    return PredictionResult(name, risk, score, v.completeness * 0.5, "Physchem heuristic", tox_color(risk))

def predict_excretion(v: DescriptorValidator) -> Dict[str, PredictionResult]:
    renal_score = 1.0
    if v.mw is not None and v.mw > 400: renal_score -= 0.4
    if v.logp is not None and v.logp > 3: renal_score -= 0.4
    if v.tpsa is not None and v.tpsa < 50: renal_score -= 0.3
    renal_score = clamp(renal_score)
    renal_lvl = PredictionLevel.HIGH if renal_score > 0.6 else PredictionLevel.LOW
    bil_score = 1.0 - renal_score
    bil_lvl = PredictionLevel.HIGH if bil_score > 0.6 else PredictionLevel.LOW
    hl_score = clamp((v.logp or 0) / 5.0)
    hl_lvl = "Long" if hl_score > 0.6 else "Short"
    return {
        "Renal Clearance": PredictionResult("Renal Clearance", renal_lvl, renal_score, v.completeness * 0.6, "Low MW/LogP favors renal", status_color(renal_lvl)),
        "Biliary Excretion": PredictionResult("Biliary Excretion", bil_lvl, bil_score, v.completeness * 0.6, "High MW/LogP favors biliary", status_color(bil_lvl)),
        "Half-Life": PredictionResult("Half-Life", hl_lvl, hl_score, v.completeness * 0.6, "Lipophilicity proxy", status_color(hl_lvl))
    }

# ── Core Engine ──────────────────────────────────────────────────────────
def compute_complete_admet(props: dict, metadata_kwargs: dict = None) -> ADMETSummary:
    v = DescriptorValidator(props)
    meta_kwargs = metadata_kwargs or {}
    metadata = Metadata(version=__version__, timestamp=datetime.utcnow().isoformat(), **meta_kwargs)
    
    dl_rules = {
        "Lipinski": eval_lipinski(v), "Veber": eval_veber(v), "Ghose": eval_ghose(v),
        "Egan": eval_egan(v), "Muegge": eval_muegge(v), "Pfizer": eval_pfizer(v), "GSK": eval_gsk(v)
    }
    
    abs_preds = {
        "GI Absorption": predict_gi_absorption(v),
        "Solubility": predict_solubility(v),
        "Bioavailability": predict_bioavailability(v, dl_rules),
        "P-gp": predict_pgp(v)
    }
    
    dist_preds = { "BBB": predict_bbb(v), "PPB": predict_ppb(v) }
    
    met_preds = {
        "CYP1A2": _cyp_heuristic(v, 300, 3.0, "CYP1A2 Inhibitor"),
        "CYP2C9": _cyp_heuristic(v, 350, 3.5, "CYP2C9 Inhibitor"),
        "CYP2C19": _cyp_heuristic(v, 350, 3.5, "CYP2C19 Inhibitor"),
        "CYP2D6": _cyp_heuristic(v, 300, 3.0, "CYP2D6 Inhibitor"),
        "CYP3A4": _cyp_heuristic(v, 400, 4.0, "CYP3A4 Inhibitor")
    }
    
    exc_preds = predict_excretion(v)
    
    herg_risk = PredictionLevel.HIGH if (v.logp is not None and v.logp > 3.7 and v.mw is not None and v.mw > 300) else PredictionLevel.LOW
    hep_risk = PredictionLevel.HIGH if (v.logp is not None and v.logp > 4 and v.rb is not None and v.rb > 10) else PredictionLevel.LOW
    tox_preds = {
        "hERG": PredictionResult("hERG Liability", herg_risk, 1.0 if herg_risk == PredictionLevel.HIGH else 0.0, v.completeness * 0.6, "MW/LogP heuristic", tox_color(herg_risk)),
        "Hepatotoxicity": PredictionResult("Hepatotoxicity", hep_risk, 1.0 if hep_risk == PredictionLevel.HIGH else 0.0, v.completeness * 0.4, "Lipophilicity/RotBonds heuristic", tox_color(hep_risk)),
        "Mutagenicity": PredictionResult("Mutagenicity", PredictionLevel.INCONCLUSIVE, 0.5, 0.1, "Requires structural alerts", "yellow"),
        "Carcinogenicity": PredictionResult("Carcinogenicity", PredictionLevel.INCONCLUSIVE, 0.5, 0.1, "Requires structural alerts", "yellow")
    }
    
    avg_dl = sum(r.score for r in dl_rules.values()) / len(dl_rules)
    avg_abs = sum(r.score for r in abs_preds.values()) / len(abs_preds)
    avg_dist = sum(r.score for r in dist_preds.values()) / len(dist_preds)
    avg_met = 1.0 - (sum(r.score for r in met_preds.values()) / len(met_preds))
    eval_tox = [tox_preds["hERG"], tox_preds["Hepatotoxicity"]]
    avg_tox = 1.0 - (sum(r.score for r in eval_tox) / len(eval_tox))

    raw_admet = ((avg_dl * SCORE_WEIGHT_DRUGLIKENESS) + (avg_abs * SCORE_WEIGHT_ABSORPTION) +
                 (avg_dist * SCORE_WEIGHT_DISTRIBUTION) + (avg_met * SCORE_WEIGHT_METABOLISM) +
                 (avg_tox * SCORE_WEIGHT_TOXICITY))
    admet_score = clamp(raw_admet) * 100
    
    if admet_score >= 85: classification = PredictionLevel.EXCELLENT
    elif admet_score >= 70: classification = PredictionLevel.VERY_GOOD
    elif admet_score >= 55: classification = PredictionLevel.GOOD
    elif admet_score >= 40: classification = PredictionLevel.MODERATE
    elif admet_score >= 25: classification = PredictionLevel.POOR
    else: classification = PredictionLevel.UNSUITABLE
    
    confidence_score = round(v.completeness * 100 - (v.missing_count * 5), 1)
    
    strengths, limitations = [], []
    if dl_rules["Lipinski"].status == Status.PASS: strengths.append("Excellent Lipinski compliance")
    else: limitations.append(f"Fails Lipinski ({dl_rules['Lipinski'].violations} violations)")
    if abs_preds["GI Absorption"].prediction == PredictionLevel.HIGH: strengths.append("High predicted oral absorption")
    else: limitations.append("Poor predicted oral absorption")
    if tox_preds["hERG"].prediction == PredictionLevel.LOW: strengths.append("Low predicted hERG liability")
    else: limitations.append("Flagged for potential hERG liability")
    
    if classification in (PredictionLevel.EXCELLENT, PredictionLevel.VERY_GOOD):
        overall = "Suitable for virtual screening and lead optimization."
    elif classification == PredictionLevel.GOOD:
        overall = "Acceptable profile, but requires targeted optimization."
    else:
        overall = "Suboptimal physicochemical profile. Heavy structural optimization needed."
    
    recommendation = StructuredRecommendation(strengths=strengths, limitations=limitations, overall_recommendation=overall)
    
    return ADMETSummary(
        metadata=metadata, drug_likeness=dl_rules, absorption=abs_preds, distribution=dist_preds,
        metabolism=met_preds, excretion=exc_preds, toxicity=tox_preds, drug_likeness_score=round(avg_dl * 100, 1),
        admet_score=round(admet_score, 1), confidence_score=clamp(confidence_score, 0, 100),
        classification=classification, recommendation=recommendation
    )

def compute_admet_flags(props: dict) -> dict:
    summary = compute_complete_admet(props)
    flags = {
        "Lipinski Ro5": (summary.drug_likeness["Lipinski"].status.value, summary.drug_likeness["Lipinski"].reason),
        "Oral Absorption": (summary.drug_likeness["Veber"].status.value, summary.drug_likeness["Veber"].reason),
        "BBB Penetration": ("PASS" if summary.distribution["BBB"].prediction == PredictionLevel.HIGH else "WARN", summary.distribution["BBB"].reason),
        "hERG Liability": ("WARN" if summary.toxicity["hERG"].prediction == PredictionLevel.HIGH else "PASS", summary.toxicity["hERG"].reason),
        "Drug-Likeness Score": ("PASS" if summary.drug_likeness_score >= 80 else ("WARN" if summary.drug_likeness_score >= 40 else "FAIL"), f"Score: {summary.drug_likeness_score}/100")
    }
    return flags

if __name__ == "__main__":
    sample = {"molecular_weight": 320, "xlogp": 2.4, "h_donors": 2, "h_acceptors": 6, "rotatable_bonds": 5, "tpsa": 74}
    import json
    summary = compute_complete_admet(sample, metadata_kwargs={"compound_name": "Test Compound"})
    print(json.dumps(summary.as_dict(), indent=2))
