"""Paths and constants for the FPL pipeline. Every magic number the workbook used lives here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(ROOT, "inputs")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
SPORTSBET_DIR = os.path.join(ROOT, "sportsbet")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COEFFICIENTS_JSON = os.path.join(DATA_DIR, "coefficients.json")
WORKBOOK = os.path.join(ROOT, "Fantasy Premier League.xlsx")
# Odds provenance manifest — which sportsbet markets are real vs synthetic right now (fpl_pipeline/provenance.py)
PROVENANCE_JSON = os.path.join(SPORTSBET_DIR, "_provenance.json")

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

# Use the trained forward-projection models (outputs/models/) for F3-F8 clean_sheet/concede2/
# saves3 instead of factor x baseline. Only these three beat the real pipeline value on
# walk-forward (2025-26); score1's F1-blend already wins, assist/yellow never do. saves3 keeps
# its PROJECTION_BLEND. See tools/save_projection_models.py + fpl_pipeline/projection_serving.py.
USE_PROJECTION_MODEL = True
# assist is NOT here: the model beats the RAW baseline (+2%) but the deployed serving blends 0.85
# with the current odds, and the blended BASELINE beats the blended MODEL (-2.1% walk-forward,
# 2026-08). The F1 anchor is what helps attacking stats, not the tree. score1 same. See memory.
PROJECTION_MODEL_STATS = {"clean_sheet": "Clean Sheet", "concede2": "Concede 2+ Goals",
                          "saves3": "3+ Saves"}

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
# Face-value DefCon prior for players with NO Premier League history — promoted-club (Championship)
# and foreign-league signings — built by tools/build_dc_prior.py from raw defensive components
# (name, team, position, dc90 proxy, minutes). Merged in load_defensive_contributions ONLY where the
# FPL PL prior has no row for that name, so a good new player starts from real evidence, not the average.
# Absent file -> no external priors (parity mode uses the workbook loader and never touches this).
EXTERNAL_DC_PRIOR = os.path.join(INPUTS_DIR, "external_dc_prior.csv")
# Improved mode: this season's minutes are weighted this many times prior-season minutes when
# blending the DC RATE (dc90) - recent form counts for more. Applies ONLY to the rate's weighted
# average (numerator AND denominator), so it stays a proper mean. It deliberately does NOT touch
# the EVIDENCE count (nineties = true minutes / 90), which drives the reliability gate and the
# shrinkage below - a recency preference is not extra evidence, so a player with one recent match
# is still one match of evidence (25% own / 75% average), not two. Set to 1.0 to disable.
DC_CURRENT_SEASON_WEIGHT = 2.0
# Improved mode: for a player who CHANGED CLUBS between seasons, his prior (old-club) minutes count
# only this fraction in the RATE blend — his old role is stale, so his dc90 leans toward this season's
# new club. Like DC_CURRENT_SEASON_WEIGHT it tilts only the RATE, NOT the evidence (nineties = true
# minutes), so a mover keeps full trust but a new-role rate. 1.0 = no mover discount; 0.5 = old club
# at half weight (≈ this season weighted 2x more, for movers only). Only bites players in BOTH seasons
# at different clubs — new/promoted players have no prior, so are untouched.
DC_MOVER_PRIOR_WEIGHT = 0.5
# Improved mode: a player's own DC RATE (dc90) is shrunk toward the reliable-population average dc90 in
# proportion to his evidence — weight = nineties / DC_SHRINK_NINETIES, capped at 1. At 4 (@360 min), a
# player needs four full matches before his own rate is fully trusted; one match = 25% own + 75% average.
# Set conservative (2026-08) once external DefCon priors backfilled promoted/foreign players (see
# external_dc_prior.csv): good new players now carry ~20 nineties of prior evidence and clear the gate
# regardless, so the gate no longer needs to run fast to surface them — it only governs whoever is BOTH
# thin AND unbackfilled, where slow = fewer single-game blow-ups. Blends the RATE then converts to
# probability. Parity keeps the workbook's hard >=4 cliff (own rate or the frozen average).
DC_SHRINK_NINETIES = 4.0
# ...but a brief cameo counts for nothing: below this many nineties (0.65 = ~59 min, roughly a
# played-most-of-the-match appearance) the weight is zero (straight population average). At or
# above it, weight = nineties/4 as above, so 0.65 nineties -> 0.65/4 own, 3.35/4 average.
DC_SHRINK_MIN_NINETIES = 0.65

# Predicted match odds (F2 fallback + F3-F8): win_pred and opp_win_pred are fitted INDEPENDENTLY,
# so the implied draw (1 - win - opp) is an unconstrained residual that balloons when both fits
# undershoot (a one-sided match could show a ~37% draw). model.reconcile_win_draw clamps the draw
# into a decisiveness-aware band before splitting the rest back to win/opp by their ratio. Band:
# draw in [DRAW_FLOOR, DRAW_CEIL_EVEN - DRAW_CEIL_SLOPE * |2r-1|]. Anchored to the 2025-26 PL draw
# rate (27.4%, roughly flat vs match closeness - only a mild taper at big mismatches); improved
# mode only, so parity is untouched. Matches already inside the band are left UNCHANGED.
DRAW_CEIL_EVEN = 0.32     # max plausible draw for an even match (league avg ~0.27, ceiling above it)
DRAW_CEIL_SLOPE = 0.14    # how the ceiling tapers with decisiveness (even 0.32 -> total mismatch 0.18)
DRAW_FLOOR = 0.15         # min plausible draw (also replaces the old sum>1 -> zero-draw guard)

# Bonus points: when True (improved mode only), replace the flat linear P(bonus) uplift with an
# odds-anchored model — each team's bonus pot from win/draw/loss probabilities, split within the
# team by XP above the appearance floor (model.bonus_points_odds). Measured to fix the ~2.2x
# under-allocation and the fodder-vs-elite mis-shape. OFF until validated on the backtest harness.
BONUS_ODDS_MODEL = True
