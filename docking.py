"""
docking.py
----------
AutoDock Vina integration backend: binary detection, running a docking
job as a subprocess, and parsing the resulting log for scores. Pure
logic only — no Streamlit widgets here. UI lives in app.py's
render_vina_tab(), which calls into this module.
"""

import os
import re
import subprocess
from typing import Optional

VINA_TIMEOUT_SECONDS = 300       # 5 minutes
VERSION_CHECK_TIMEOUT = 5


def is_vina_available() -> bool:
    """Check if the AutoDock Vina binary is available in PATH."""
    try:
        result = subprocess.run(
            ["vina", "--version"],
            capture_output=True, text=True, timeout=VERSION_CHECK_TIMEOUT,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_vina_command(
    receptor_path: str,
    ligand_path: str,
    out_path: str,
    log_path: str,
    center: tuple[float, float, float],
    box_size: float,
    exhaustiveness: int,
) -> list[str]:
    """Build the Vina CLI command for a docking run."""
    cx, cy, cz = center
    return [
        "vina",
        "--receptor", receptor_path, "--ligand", ligand_path,
        "--out", out_path, "--log", log_path,
        f"--center_x={cx}", f"--center_y={cy}", f"--center_z={cz}",
        f"--size_x={box_size}", f"--size_y={box_size}", f"--size_z={box_size}",
        f"--exhaustiveness={exhaustiveness}",
    ]


def run_vina_docking(
    receptor_path: str,
    ligand_path: str,
    out_path: str,
    log_path: str,
    center: tuple[float, float, float],
    box_size: float,
    exhaustiveness: int,
) -> dict:
    """
    Run a real AutoDock Vina docking job as a subprocess.

    Returns a result dict:
        {"success": bool, "log_text": str | None, "best_score": float | None,
         "error": str | None}
    """
    cmd = build_vina_command(receptor_path, ligand_path, out_path, log_path,
                              center, box_size, exhaustiveness)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=VINA_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Vina timed out after 5 minutes.",
                "log_text": None, "best_score": None}
    except Exception as e:
        return {"success": False, "error": f"Error running Vina: {e}",
                "log_text": None, "best_score": None}

    if proc.returncode != 0:
        return {"success": False, "error": f"Vina error: {proc.stderr}",
                "log_text": None, "best_score": None}

    log_text = None
    best_score = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_text = f.read()
        best_score = parse_best_score(log_text)

    return {"success": True, "error": None, "log_text": log_text, "best_score": best_score}


def parse_best_score(log_text: str) -> Optional[float]:
    """Extract the top-ranked binding affinity (kcal/mol) from a Vina log."""
    scores = re.findall(r"\s+1\s+([-\d.]+)", log_text)
    return float(scores[0]) if scores else None
