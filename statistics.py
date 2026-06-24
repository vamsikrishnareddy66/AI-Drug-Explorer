"""
statistics.py
-------------
Summary statistics for virtual screening results.
"""

import pandas as pd


def calculate_statistics(results_df: pd.DataFrame) -> dict:
    if results_df.empty:
        return {"count": 0, "best_score": None, "worst_score": None, "average_score": None, "avg_mw": None}

    score_col = "simulated_score" if "simulated_score" in results_df.columns else "docking_score"
    scores = results_df[score_col]
    avg_mw = round(results_df["molecular_weight"].mean(), 2) if "molecular_weight" in results_df.columns else None

    return {
        "count":         len(results_df),
        "best_score":    round(scores.min(), 2),
        "worst_score":   round(scores.max(), 2),
        "average_score": round(scores.mean(), 2),
        "avg_mw":        avg_mw,
    }
