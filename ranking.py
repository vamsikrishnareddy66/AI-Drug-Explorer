"""
ranking.py
----------
Ranks compounds by simulated score (ascending — lower = better).
Uses 'simulated_score' column; falls back to 'docking_score' for
backwards compatibility.
"""

import pandas as pd


def rank_compounds(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    score_col = "simulated_score" if "simulated_score" in results_df.columns else "docking_score"
    ranked = results_df.sort_values(by=score_col, ascending=True).copy()
    ranked.reset_index(drop=True, inplace=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def get_best_compound(ranked_df: pd.DataFrame) -> dict:
    if ranked_df.empty:
        return None
    return ranked_df.iloc[0].to_dict()
