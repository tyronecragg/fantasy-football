"""Paths and constants for the FPL pipeline. Every magic number the workbook used lives here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(ROOT, "inputs")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
SPORTSBET_DIR = os.path.join(ROOT, "sportsbet")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COEFFICIENTS_JSON = os.path.join(DATA_DIR, "coefficients.json")
WORKBOOK = os.path.join(ROOT, "Fantasy Premier League.xlsx")

SEASON = "2026-2027"
FPL_DATA_DIR = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", SEASON)

# The frozen workbook (parity reference) is a 2025-2026 artifact; parity mode pins its
# season and reads inputs/odds from a frozen snapshot so --validate keeps comparing
# like-for-like after season rollovers and weekly scrapes.
PARITY_SEASON = "2025-2026"
PARITY_INPUTS_DIR = os.path.join(ROOT, "parity_reference", "inputs")
PARITY_SPORTSBET_DIR = os.path.join(ROOT, "parity_reference", "sportsbet")

# Bookmaker margin divisors (odds -> probability = 1/odds/margin)
MARGIN_PLAYER = 1.05   # player & team match markets
MARGIN_CARD = 1.10     # Bet365 "player to be booked" — higher overround than goals/assists

# 2+ assists as a fraction of the 1+ chance, measured on the CALIBRATED probabilities the fill
# is applied to (i.e. after MARGIN_PLAYER and the longshot shrink, not the raw odds). Real 2+
# assists sit at a roughly FLAT ~0.09x the 1+ chance in that space, while the Poisson tail used
# for 2+ goals (model.poisson_score2) has the wrong shape — it ramps ~0.03->0.15 with the 1+
# chance. The flat ratio fits better (2026-08-19, n=187 priced in both: mean abs error 0.0058
# vs Poisson's 0.0081). So missing 2+ assists are filled with this ratio, not the curve;
# players.py re-measures it each run and falls back to this default only when too few pairs
# exist. NOTE: on the RAW de-margined odds the same ratio is ~0.24 - the gap is the longshot
# calibration shrinking the small 2+ numbers, which is provisional for 2+ markets (see #25).
ASSIST2_RATIO = 0.088

# Longshot calibration for PLAYER attacking markets (goalscorer / 2+ goals / assists).
# A flat MARGIN_PLAYER cannot be right: the total load on Betway's goalscorer market
# measures ~3.5x (tools/margin_goals.py) while the favourite's price is already near fair,
# so the load sits on the longshots. Measured against outcomes (tools/calibration.py,
# 2025-26 GW16-29, minutes>=60) the pipeline's own probabilities came out:
#     25-35% -> 27.3% actual (1.08x)     8-12% -> 4.2% (2.38x)
#     18-25% -> 18.7%        (1.14x)      5-8%  -> 2.8% (2.42x)
#     12-18% -> 10.3%        (1.46x)      2-5%  -> 0.8% (5.23x)
# Favourites are calibrated; longshots are overstated 2.4-5x. Knots are (probability,
# multiplier), interpolated in log-probability space and clamped monotone so ranking is
# preserved. Set to None to disable.
#
# PROVISIONAL — see the "Longshot calibration" section of README.md for what is still
# outstanding before this should be trusted for anything but experimentation.
LONGSHOT_FLOOR = 0.15        # never shrink below this; a fitted 0.00 would say "impossible"
LONGSHOT_CALIBRATION = [
    (0.04, 0.20),
    (0.065, 0.41),
    (0.10, 0.42),
    (0.15, 0.68),
    (0.21, 0.88),
    (0.29, 0.93),
    (0.42, 0.96),
]
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

# Stats whose live factor is the trailing MEDIAN of this season's weekly factors
# (factor experiment 2026-08: median beat single-week on holdout for all of these —
# yellow by 14%, concede2 7%, score1/saves3 5%, clean_sheet 4%, concede4/saves6
# confirmed; ASSIST was consistently WORSE with median in both splits — role changes
# outpace smoothing — so it stays single-week).
MEDIAN_FACTOR_STATS = ("score1", "yellow", "clean_sheet", "concede2", "concede4",
                       "saves3", "saves6")

POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]

N_FIXTURES = 6

# Defensive-contribution blending: prior-season DC-per-90 is blended with the current
# season minutes-weighted, with the prior's weight capped at this many minutes (19 full
# matches = half a season) — most regulars then carry their whole last-season DC sample
# (median ~15 nineties), only ever-presents are trimmed. Pure prior at GW1, ~50/50 after
# 19 current matches, current-dominated in the run-in. DefCon has only one prior season
# (2025-26; the stat is new), so this is the deepest history available.
DC_PRIOR_CAP_MINUTES = 1710
# Improved mode: a player's own DC hit-probability is shrunk toward the reliable-population
# average in proportion to his evidence — weight = nineties / DC_SHRINK_NINETIES, capped at 1. So one
# full match = 25% his own + 75% the average; four or more = his own (minimum below). Parity keeps
# the workbook's hard >=4 cliff (own rate or the average, nothing in between).
DC_SHRINK_NINETIES = 4.0
# ...but a brief cameo counts for nothing: below this many nineties (0.65 = ~59 min, roughly a
# played-most-of-the-match appearance) the weight is zero (straight population average). At or
# above it, weight = nineties/4 as above, so 0.65 nineties -> 0.65/4 own, 3.35/4 average.
DC_SHRINK_MIN_NINETIES = 0.65
