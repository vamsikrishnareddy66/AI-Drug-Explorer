"""
AI Drug Discovery Platform

A modular Python package for educational virtual screening,
protein analysis, compound ranking, statistical analysis,
and scientific report generation.

Author: Vamsi Krishna Reddy
Project: B.Tech Biotechnology Final Year Project
Year: 2026
"""

__version__ = "1.0.0"
__author__ = "Vamsi Krishna Reddy"
__email__ = ""
__license__ = "All Rights Reserved"

# Core modules
from . import protein
from . import screening
from . import ranking
from . import statistics
from . import report

__all__ = [
    "protein",
    "screening",
    "ranking",
    "statistics",
    "report",
]

PACKAGE_NAME = "AI Drug Discovery Platform"

DESCRIPTION = (
    "Educational platform for protein analysis, virtual screening, "
    "compound ranking, statistical visualization, and scientific "
    "report generation."
)
