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
# Booking odds -> points calibration (improved mode), applied where a yellow probability becomes points so every
# horizon, the archive and the trained models stay on the market's scale. 2026-27 GW1-6 (1,245 player-games of
# 60+ min, 188 booked): the pipeline's F1 yellow ran 1.20x the actual rate, over in EVERY week (1.09-1.45x);
# x0.85 had the best Brier (0.12537 vs 0.12626, player bootstrap P = 0.90). Measured 2026-10-17; re-check ~GW12.
YELLOW_CALIBRATION = 0.85

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
# ASSISTS get their own, gentler curve (2026-09-09): the knots above were fitted on Sportsbet
# goalscorer outcomes and applied to every attacking market. On Betway this season (GW1-3,
# minutes>=60, tools/calibration.py) goals pool at 1.06x but assists at 0.69x, with the two
# lowest bands projecting 13 assists and getting 28 (~4 s.e.); the 1xBet two-way cross-check
# put Betway's raw assist prices within 8% of a de-margined book. Half strength in log space
# (multipliers square-rooted) lifts those bands to ~0.8x of actual. Re-fit ~GW9 on outcomes.
# 2026-09-21, five gameweeks (924 player-games of 60+ min with assist odds, 101 assisters): the half-
# strength curve was still 16% under (0.84x); quarter strength 0.94x; the RAW de-margined Betway price
# 1.07x with the best Brier. The only band raw over-states is the very bottom (<6%: 1.23x), so the assist
# curve is now a light trim there and nothing above 8%: pooled 1.05x, bands 1.03 / 0.88 / 1.01 / 1.19 /
# 1.03. (The >30% band reads 1.7x over on 14 rows / 3 assists - too thin to act on.) The old half curve
# was: [(p, round(m ** 0.5, 3)) for p, m in LONGSHOT_CALIBRATION]. Re-check ~GW9.
LONGSHOT_CALIBRATION_ASSIST = [(0.04, 0.80), (0.08, 1.00)]
# The ARCHIVE stores each week's F1 Assist as calibrated AT THE TIME, and the forward assist factor is a
# trailing median over those weeks (history.season_weekly_factors) - so every change of curve would leave
# the median anchored to the old regime for the rest of the season. The archive is never rewritten;
# instead each archived week is re-expressed under the CURRENT curve when the factors are read: invert
# the curve that week was archived under, apply today's. (season, first GW, last GW, knots-at-the-time).
# Weeks not listed were archived under the current curve. Add a row whenever the assist curve changes.
ASSIST_CURVE_REGIMES = [
    ("2026-2027", 1, 3, LONGSHOT_CALIBRATION),                                             # full goals curve
    ("2026-2027", 4, 5, [(p, round(m ** 0.5, 3)) for p, m in LONGSHOT_CALIBRATION]),      # half strength
]
MARGIN_WDW = 1.03      # win-draw-win
MARGIN_SEASON = 1.08   # title / relegation / top-6

# "No market" sentinels used by the Overall Odds sheet when average odds are 0
SENTINEL_TITLE_TOP6 = 5001
SENTINEL_RELEGATION = 2001

# Goalkeeper-saves defaults when no odds exist (IFERROR fallbacks)
SAVES3_DEFAULT = 0.6
SAVES6_DEFAULT = 0.0

# Projection blend (improved mode): modelled F2-F8 probability = w * model + (1-w) * the
# player's current F1 odds-implied probability. Stats absent here are pure model.
# EMPTY since 2026-10-23 (Tyrone): was {"score1": .70, "assist": .85, "saves3": .85}, fitted on
# 2025-26 GW16-29 for average error vs the later real odds. Blending the RAW current price carries
# this week's fixture into every future week (tough fixtures projected too high, soft too low).
# Re-tested on RANKING (within-position rank match + top-5 pick shortfall, the transfer decision):
# on 2026-27 Betway data no blend ranks and picks best for goals and keepers, assists ~level;
# last season (Sportsbet) preferred a last-3-week price average. Re-test ~GW12 (docs/TODO.md).
PROJECTION_BLEND = {}

# Use the trained forward-projection models (outputs/models/) for F3-F8 clean_sheet/concede2/
# saves3 instead of factor x baseline. Only these three beat the real pipeline value on
# walk-forward (2025-26); score1's F1-blend already wins, assist/yellow never do. saves3 keeps
# its PROJECTION_BLEND. See tools/save_projection_models.py + fpl_pipeline/projection_serving.py.
USE_PROJECTION_MODEL = True

# Rolling odds-implied team rating as the F3-F8 (and F2-fallback) win / opponent-win source,
# replacing the season-outrights Lasso (model.win_pred) in improved mode. fpl_pipeline/rating.py;
# tested 2026-09-02 (tools/rolling_rating.py): match-odds MAE 0.045 vs 0.052 outrights / 0.066
# static rating, 0.025 vs 0.035 on live GW4 prices. Falls back to the outrights model when the
# maps or football-data prices are unavailable. The outrights still feed the GBM features, the
# fixture-outlook tool and chip planning.
USE_ROLLING_RATING = True
RATING_HALFLIFE_DAYS = 30.0        # a match's weight halves every 30 days (grid-best; 20-45 similar)
RATING_PROMOTED_WEIGHT = 2.0       # promoted side's prior = 2 matches' worth at the three-lowest level
RATING_MAPS_JSON = os.path.join(DATA_DIR, "rating_maps.json")   # tools/fit_rating_maps.py
# Personal team views (inputs/team_views.csv: Gameweek,Team,nudge,note). A nudge is "this team is
# n goals per game better (+) / worse (-) than the rating says". Only rows whose Gameweek equals the
# window's current F1 gameweek apply, so a forgotten row goes inert at the next roll. Applied to the
# rolling rating (full strength, F3-F8 + unpriced F2) and the priced weeks' bookmaker numbers
# (fpl_pipeline/team_views.py, at TEAM_VIEW_MARKET_WEIGHT). USE_TEAM_VIEWS=False to compare.

# Fixture risk (inputs/fixture_risk.csv, current F1 gameweek only): chance a match is played. Each listed
# team's F1 XP is scaled by it in the published master; the archive keeps the unscaled numbers.
USE_FIXTURE_RISK = True
FIXTURE_RISK_CSV = os.path.join(INPUTS_DIR, "fixture_risk.csv")
USE_TEAM_VIEWS = True
TEAM_VIEWS_CSV = os.path.join(INPUTS_DIR, "team_views.csv")
TEAM_VIEW_LOGODDS_PER_GOAL = 1.3   # log(pH/pA) per goal of supremacy (Poisson, typical PL scorelines)
TEAM_VIEW_AVG_GOALS = 1.35         # league-average goals per team per game (converts goals -> log rate)
TEAM_VIEW_MAX = 0.5                # |nudge| cap in goals per game - a guard against typos
TEAM_VIEW_MARKET_WEIGHT = 0.5      # strength on the PRICED weeks (F1, priced F2): 0 = leave Betway alone, 1 = full
# assist is NOT here: the model beats the RAW baseline (+2%) but the deployed serving blends 0.85
# with the current odds, and the blended BASELINE beats the blended MODEL (-2.1% walk-forward,
# 2026-08). The F1 anchor is what helps attacking stats, not the tree. score1 same. See memory.
PROJECTION_MODEL_STATS = {"clean_sheet": "Clean Sheet", "concede2": "Concede 2+ Goals",
                          "saves3": "3+ Saves"}
# 2026-10-17 refresh: the defensive GBMs are served as w*model + (1-w)*raw factor x baseline instead of
# pure model. Three one-week leak-free folds (2026-27 gw_to 5-7, models retrained on 2025-26 + earlier 2026-27):
# clean_sheet pure 0.0251 -> 0.0233 at w=0.75 (-7%); concede2 0.0340 -> 0.0309 at 0.65 (-9%); saves3 0.0652 ->
# 0.0631 at 0.80 (-3%). Stats absent here are served pure. Re-fit with the models.
DEFENSIVE_MODEL_WEIGHT = {"clean_sheet": 0.75, "concede2": 0.65, "saves3": 0.80}
# STACKED stats (2026-09-02): the GBM does not beat the F1-blend on its own for goals / yellow, but
# w*model + (1-w)*blend does - like-for-like walk-forward with player bootstrap
# (tools/assist_walkforward.py, set E: rating win/opp + expected goals + the baseline as a feature +
# 6/10-week odds form): goals stack -4.5% error (P=1.00, and -6.6% on the 2026-27 cross-season fold),
# yellow stack -7.1% (P=1.00). Weights = the VAL-fitted stack weights; re-checked 2026-10-17 on 2026-27 folds
# after the retrain (score1 best 0.65, yellow flat 0.60-0.70) - unchanged. Assists are NOT stacked: the
# in-season gain (~+2.5%) reversed on the cross-season fold (-4% to -18%).
PROJECTION_STACK = {"score1": ("Score 1+", 0.65), "yellow": ("Yellow Card", 0.70)}
# A stack is only applied once the season archive holds this many real weeks (yellow leans on
# 10-week odds form and had no cross-season fold to prove itself; goals passed that fold, so 0).
PROJECTION_STACK_MIN_WEEKS = {"score1": 0, "yellow": 6}

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

# Projection horizon (improved mode): how many fixtures F1..HORIZON the pipeline projects. F1/F2 use
# real odds; F3..HORIZON are fully model-driven (win_pred/rating + factor x baseline + GBM). Set to 17
# so a GW3 run reaches GW19 (first half) for chip-window planning. Beyond F8 there's no team news, so
# start-probs are held flat at the F8 belief; the GBM's `horizon` feature saturates at its trained max
# (F8), so its far-out horizon-decay plateaus while the per-fixture (opponent/venue/rating) signal
# still varies. The OPTIMISER keeps its own num_fixtures (default 8) and the ARCHIVE stays F1-F8
# (history.py only writes columns already in historical_player_data.csv), so both are unaffected.
HORIZON = 17

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
# Improved mode: MINUTES PER START. dc90 is a per-90 rate, so dc_probability is P(hit | plays 90) - but
# 46% of midfield starts and 21% of defender starts end before 90, and a midfielder subbed at 60-74
# hits the 12 threshold 2% of the time against ~20% priced (2025-26 + 2026-27 starts, measured
# 2026-09-16). P(hit | started) was 0.75x the model for DEF and 0.67x for MID; full-90 starters were
# calibrated. So the rate is scaled by the player's expected minutes per start / 90 before it becomes
# a probability. Minutes per start = his starts this season (weighted DC_CURRENT_SEASON_WEIGHT x, like
# the rate) + up to DC_MINUTES_PRIOR_CAP_STARTS from last season (prior-club starts at
# DC_MOVER_PRIOR_WEIGHT for movers, like the rate), shrunk toward the position average with
# DC_MINUTES_SHRINK_STARTS pseudo-starts. At 7 pseudo-starts and 2x recency a new signing is 53% his own
# number after four starts, 67% after seven; a full prior season 59% at GW1 and 81% after ten starts
# (calibration slope ~0.85 no-prior / ~1.05 with-prior on 2025-27 starts; 5 over-trusted new signings). Position
# averages over all 2025-26 starts (DEF 85.2, MID 80.1, FWD 78.5; 2026-27 to date 84.7 / 78.5 / 79.4).
# False = price the full 90.
DC_MINUTES_PER_START = True
DC_MINUTES_POSITION_AVG = {"DEF": 85.0, "MID": 80.0, "FWD": 79.0}
# 2026-09-21: the minutes enter as a MIXTURE, not an average. Plugging in average minutes (rate x mps/90)
# over-corrected midfielders (walk-forward, 17 GWs / 3,013 starts, tools-style as-of rebuild: MID
# predicted/actual 1.21 flat -> 0.86) because a player who plays 90 half the time and ~70 the rest clears
# the threshold more often than one who always plays 80. So P(hit | start) = p90 x P(hit | full rate)
# + (1 - p90) x P(hit | rate x sub_minutes / 90): MID Brier 0.1163 -> 0.1152 (ratio 0.89), DEF unchanged.
# p90 = the player's share of starts that go the full 90, with the same evidence rules as minutes per
# start (this season 2x, <=10 prior starts, mover discount, DC_MINUTES_SHRINK_STARTS pseudo-starts at the
# position average). Sub minutes = mean minutes of starts that end early (2025-26: DEF 67.6, MID 69.6,
# FWD 70.8). `mps` is still computed and written to the dc_stats outputs for reference.
DC_FULL90_POSITION_AVG = {"DEF": 0.78, "MID": 0.49, "FWD": 0.42}
DC_SUB_MINUTES = {"DEF": 68.0, "MID": 70.0, "FWD": 70.5}
DC_MINUTES_SHRINK_STARTS = 7
DC_MINUTES_PRIOR_CAP_STARTS = 10
# Improved mode: HEAVY-FAVOURITE DISCOUNT, per fixture. A defender's DC rate in games where his team is
# a 70%+ favourite runs x0.82 his own norm (player fixed effects, home + minutes controlled, 2025-26 +
# 2026-27, 1,354 DEF starts / 55 heavy: t = -3.1, within-player permutation p = 0.007; replicates in
# 2025-26 alone; 0.55-0.70 band x0.93). Midfielders: x0.91, t = -1.8, p = 0.14 - applied lightly from
# 2026-10-17 (see below), re-test ~GW12. Position -> (win prob where the discount starts, win prob where it is fully on, multiplier
# there); linear between, 1.0 below the start, clamped above the end. The multiplier scales the RATE
# (after minutes per start) for EACH fixture from that fixture's win probability (F1/F2 odds,
# F3+ predicted), so DefCon is no longer a flat copy of F1 across the horizon. Measured 2026-09-16.
# 2026-09-21 re-fit in the walk-forward harness WITH the neutralisation below: the first curve
# (0.55 -> 0.82) left favourites over-predicted (0.55-0.70 band 1.11x, >0.70 band 3.3x); starting at
# 0.50 and ending at 0.65 calibrates <=0.55 at 1.00x and 0.55-0.70 at 1.03x with the best Brier among
# neutralised variants (0.1730 vs 0.1735). The >0.70 band is still ~2.5x over on 54 starts / 3 hits -
# too thin to chase; re-fit ~GW12.
# MID added 2026-10-17 (Tyrone: "Let's turn it on"). Walk-forward (2025-26 GW16-29 + 2026-27 GW2-6, 1,679 MID
# starts, neutralised, mixture minutes): every curve within 0.0002 Brier of off; 0.55->0.91 (the Sept measured
# x0.91, starting later so the 0.55-0.70 band - already under-predicted 0.78x - is barely touched) has the best
# Brier (0.1160 vs 0.1162) and cuts the >0.70 over-prediction 2.35x -> 1.99x. That band is only 63 starts /
# 4 hits, so this is a light touch, not a fit - re-fit ~GW12 with the DEF curve.
DC_FAVOURITE_CURVE = {"DEF": (0.50, 0.85, 0.65), "MID": (0.55, 0.85, 0.91)}
# ...and the HISTORIC rate is neutralised for the fixtures it was earned in (Tyrone, 2026-09-21). The
# x0.82 was measured against a player's OTHER games, but a City/Arsenal defender's season dc90 is
# already mostly favourite games - applying the discount to that average double-counts (a player who
# averaged 15 at a typical 80% win prob should be priced at ~15 in an 80% game, not below it). So each
# season's dc90 is divided by his team's AVERAGE multiplier over the matches played that season
# (football-data closing odds, the rolling rating's match frame) before the seasons are blended; the
# upcoming fixture's multiplier is then applied to that neutral-fixture rate. Prior-season rates use the
# prior club. No PL matches for a team (Championship/foreign priors) -> 1.0. False = no neutralisation.
DC_NEUTRALISE_FIXTURE_MIX = True

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
# KEEPERS get a flat expected bonus per start, taken off the team pot before the outfielders share
# it. Measured 2026-09-02 (tools/bonus_check.py, 2025-26 GW16-29): keeper bonus is 0.32 per start
# and NOTHING we price predicts it - expected saves has a zero coefficient, clean-sheet chance a
# slightly negative one, and the XP weight had it BACKWARDS (favourites' keepers 0.68 expected vs
# 0.26 actual, underdogs' 0.13 vs 0.47: saves earn BPS, scorers outrank a quiet keeper). A flat
# rate is the best out-of-sample model with 259 keeper rows; re-fit ~GW12.
BONUS_GK_PER_START = 0.32
# Outfield: per-position multiplier on the bonus-worthiness weight (XP above the appearance floor)
# before the remaining pot is shared - relative weights only, fitted leave-one-GW-out so each
# position's expected bonus matches its actual total (BPS pays midfielders' XP less than forwards').
# Without it DEF expected 0.24 vs actual 0.17, MID 0.38 vs 0.28, FWD 0.36 vs 0.43.
BONUS_POSITION_WEIGHT = {"DEF": 0.56, "MID": 0.61, "FWD": 1.0}
