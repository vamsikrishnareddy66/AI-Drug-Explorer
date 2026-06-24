"""
screening.py
------------
Virtual Screening Simulation module.

Renamed from "docking simulation" to "virtual screening simulation" to
better reflect that the scores are formula-based, not actual AutoDock Vina
results. Scores depend on both compound properties and protein metadata for
reproducible, realistic-feeling variation.
"""

import random
import pandas as pd


def load_compound_data(csv_path: str = "data/compounds.csv") -> pd.DataFrame:
    """Load the compound dataset with chemical properties."""
    return pd.read_csv(csv_path)


def lipinski_status(row) -> str:
    """
    Check Lipinski Rule of Five:
      MW ≤ 500 g/mol
      LogP ≤ 5
      H-bond donors ≤ 5
      H-bond acceptors ≤ 10
    Returns 'Pass', 'Fail (N violations)', or 'Borderline'.
    """
    violations = 0
    if row.get("molecular_weight", 0) > 500:
        violations += 1
    if row.get("logp", 0) > 5:
        violations += 1
    if row.get("h_donors", 0) > 5:
        violations += 1
    if row.get("h_acceptors", 0) > 10:
        violations += 1
    if violations == 0:
        return "✅ Pass"
    elif violations == 1:
        return f"⚠️ Borderline (1 violation)"
    else:
        return f"❌ Fail ({violations} violations)"


def simulate_virtual_screening(
    df: pd.DataFrame,
    pdb_id: str,
    selected_compounds: list,
    protein_info: dict = None,
) -> pd.DataFrame:
    """
    Simulate virtual screening scores for selected compounds.

    NOTE: These scores are EDUCATIONAL SIMULATIONS derived from compound
    properties (LogP, MW) and protein metadata (resolution, entity count).
    They do NOT represent actual AutoDock Vina or experimental docking results.

    Returns DataFrame with 'simulated_score' column (kcal/mol equivalent).
    Lower (more negative) = stronger simulated binding.
    """
    filtered = df[df["compound_name"].isin(selected_compounds)].copy()
    if filtered.empty:
        return filtered

    protein_info = protein_info or {}
    resolution   = protein_info.get("resolution") or 2.0
    num_entities = protein_info.get("num_entities") or 1

    seed = sum(ord(ch) for ch in pdb_id.upper())
    rng  = random.Random(seed)

    protein_shift    = rng.uniform(-1.5, 1.5)
    resolution_bonus = -0.3 if resolution <= 1.5 else (0.2 if resolution >= 3.0 else 0.0)
    entity_bonus     = -0.1 * (num_entities - 1)

    simulated_scores = []
    for _, row in filtered.iterrows():
        base_score = row["docking_score"]
        logp       = row.get("logp", 0)
        logp_factor = -0.15 if 2.0 <= logp <= 4.0 else 0.05
        compound_seed = seed + sum(ord(ch) for ch in str(row["compound_name"]))
        crng          = random.Random(compound_seed)
        pose_noise    = crng.uniform(-0.3, 0.3)
        final_score   = (base_score + protein_shift + resolution_bonus
                         + entity_bonus + logp_factor + pose_noise)
        simulated_scores.append(round(final_score, 2))

    filtered["simulated_score"] = simulated_scores
    # Keep original docking_score for reference; add lipinski
    filtered["lipinski_status"] = filtered.apply(lipinski_status, axis=1)
    filtered["protein_target"]  = pdb_id.upper()
    return filtered.reset_index(drop=True)


# Backwards-compatible alias used by any code that still calls simulate_docking
def simulate_docking(df, pdb_id, selected_compounds, protein_info=None):
    result = simulate_virtual_screening(df, pdb_id, selected_compounds, protein_info)
    if not result.empty and "simulated_score" in result.columns:
        result["docking_score"] = result["simulated_score"]
    return result
