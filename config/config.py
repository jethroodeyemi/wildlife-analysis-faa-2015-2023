"""
Configuration settings for the FAA Wildlife Strikes Analysis Project
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Data file paths
RAW_DATA_PATH = DATA_DIR / "FAA_Wildlife_Strikes_Data_03-30-2025.csv"

# Analysis parameters
YEAR_RANGE = (1990, 2023)
SIGNIFICANCE_LEVEL = 0.05

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_DPI = 300
FIGURE_SIZE_DEFAULT = (12, 8)
FIGURE_SIZE_WIDE = (15, 8)

# Color schemes
COLOR_PALETTE = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'accent': '#3498DB',
    'neutral': '#95A5A6'
}

# Time-based analysis parameters
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# Reporting parameters
MAX_ROWS_DISPLAY = 20
DECIMAL_PLACES = 2