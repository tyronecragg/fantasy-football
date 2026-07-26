"""Paths and constants for the FPL pipeline. Every magic number the workbook used lives here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(ROOT, "inputs")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
SPORTSBET_DIR = os.path.join(ROOT, "sportsbet")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COEFFICIENTS_JSON = os.path.join(DATA_DIR, "coefficients.json")
WORKBOOK = os.path.join(ROOT, "Fantasy Premier League.xlsx")

SEASON = "2025-2026"
FPL_DATA_DIR = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", SEASON)

# Bookmaker margin divisors (odds -> probability = 1/odds/margin)
MARGIN_PLAYER = 1.05   # player & team match markets
MARGIN_WDW = 1.03      # win-draw-win
MARGIN_SEASON = 1.08   # title / relegation / top-6

# "No market" sentinels used by the Overall Odds sheet when average odds are 0
SENTINEL_TITLE_TOP6 = 5001
SENTINEL_RELEGATION = 2001

# Goalkeeper-saves defaults when no odds exist (IFERROR fallbacks)
SAVES3_DEFAULT = 0.6
SAVES6_DEFAULT = 0.0

# Total XP = sum(weight_k * F_k XP)
TOTAL_XP_WEIGHTS = [1.0, 0.85, 0.7, 0.7, 0.7, 0.7]

# Projection blend (improved mode): modelled F2-F6 probability = w * model + (1-w) * the
# player's current F1 odds-implied probability. Weights fitted per stat by
# tools/backtest_projections.py on 2025-26 GW16-29 forecast-vs-actual pairs; stats
# absent here backtested best at w=1.0 (pure model) and are not blended.
PROJECTION_BLEND = {"score1": 0.70, "assist": 0.85, "saves3": 0.85}

POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]

N_FIXTURES = 6
