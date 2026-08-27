# -*- coding: utf-8 -*-
"""Build a supervised training matrix for the FORWARD-PROJECTION models — predict a future
gameweek's scoring component from what we know now (F1 data + outright odds + trailing form).

    env/Scripts/python tools/build_training_data.py

One CSV per component in outputs/training/train_<stat>.csv. Each row is a (player, gameweek M,
horizon k) example built from inputs/historical_player_data.csv (reusing the tested M->M+k pairing
in tools/backtest_projections.py):

  TARGET   actual      = the real odds-derived F1 <stat> at gameweek M+k (what actually materialised)
  BASELINE predicted   = the current pipeline's forward projection (factor x baseline) — the number to beat
  FEATURES (all known at M):
    persistence   the player's own F1 <stat> at M (the current odds-level; unchanged = persistence forecast)
    form{w}/form_n{w}  mean of the player's F1 <stat> over the EXACT prior w gameweeks (w = 3 and 10),
                  and how many of those slots existed — a gap is NOT skipped to reach further back
    momentum      persistence - form3 (is he above/below recent form?)
    form_win{w}/form_loss{w}  team's mean F1 win / opponent-win over the same w gameweeks — how easy/
                  hard the fixtures the form was earned against were (context for form{w})
    difficulty_shift  win_hat - form_win3: the upcoming fixture easier(+)/harder(-) than the form window
    peer_avg / rel_peer / rel_peer{w}  objective difficulty-adjustment: the player's F1 <stat>
                  relative to his priced same-position teammates (self-excluded), now and trailing
                  over w gameweeks — replaces the old model-derived `factor`
    win_hat/opp_hat  predicted win / opponent-win for the M+k fixture (from outright odds + schedule)
    own_/opp_ title/releg/top6   both teams' season-strength probabilities as seen at M
    *_diff        title/releg/top6/strength diffs (own - opp) — relative strength, pre-computed
    position, venue (of the M+k fixture), horizon k

So a learned model fit on these is directly comparable to `backtest_projections` (same rows, same
`predicted`/`actual`), and improving it improves the F2-F8 projections the optimiser consumes.
"""
import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.backtest_projections import STATS, build_pairs, load_archive  # noqa: E402

WINDOWS = (3,)   # ODDS-form horizon(s). Kept to 3 so a 3-week train/val embargo can fully
                 # de-correlate the walk-forward selector (a 10-GW window would need ~10; infeasible
                 # on ~12 weeks of data). Re-add 10 here to restore long odds-form context features.
REAL_WINDOWS = (3, 10)   # REAL on-pitch trailing gets its own window set (incl. 10): the market's
                 # weekly odds move little week-to-week (hence the embargo pressure), but real
                 # performance is noisier and a longer window may carry more signal worth testing.
XG_WINDOWS = (3, 6)      # trailing EXPECTED goals/assists (continuous; smoother than the binary
                 # real_form outcomes). Gated by env XG_FEATURES=1 so the with/without walk-forward
                 # isolates exactly the xG/xA contribution. TESTED 2026-08-27, OFF by default (no gain):
                 # goals +10.3% -> +10.7% (noise; model already wins, odds price finishing form),
                 # assists -1.8% -> -1.1% (best case as predicted, but STILL loses to baseline). Even
                 # continuous xG/xA can't beat the market odds - same lesson as real_form. Left gated
                 # for easy re-test as data grows; do not enable without a fresh walk-forward win.
_XG = os.environ.get("XG_FEATURES") == "1"
# env PRED_FEATURE=1: add the pipeline baseline projection `predicted` (= factor x baseline, the
# very number the model is scored against) AS A FEATURE. Turns the model into a correction on top
# of the baseline — it can reproduce it (>= parity) and only deviate where features add signal.
# The strongest lever for assists, where the model currently LOSES to that baseline (see walk-forward).
_PRED = os.environ.get("PRED_FEATURE") == "1"
# env WINHAT_FEATURES=1: add win_hat/opp_hat (the win_pred / opp_win_pred outputs for the M+k
# fixture) AS FEATURES. Historically excluded on the argument that the raw outright odds + diffs
# carry the same info objectively. TESTED 2026-08-27 (walk-forward, workbook win_pred), OFF by
# default — NO GAIN: assist -1.8% -> -0.2% (helps, but STILL loses to baseline), and it DEGRADES
# the winners: score1 +10.3% -> +9.6%, concede2 +17.3% -> +16.8%, yellow +1.8% -> +1.6%;
# clean_sheet/saves3 flat. Confirms the original rationale — the odds already price the win in, so
# feeding the model its own win_pred output is redundant and marginally harmful. (The Lasso vs
# workbook win_pred differ ~0.9pp win-prob, far too small to flip this; a Lasso-win_hat re-test
# would land in the same place.) Left gated for easy re-test as data grows.
_WINHAT = os.environ.get("WINHAT_FEATURES") == "1"
# env BLEND_FEATURE=1: add the fitted F1-blend (w*predicted + (1-w)*persistence, w=PROJECTION_BLEND
# per stat) AS A FEATURE — the smooth linear combo the tree can't reproduce from predicted+persistence
# separately. TESTED 2026-08 (+ widened 60-trial search): still NO GAIN. score1 model 0.0232 vs the
# pure blend 0.0229 (-1.2%); and for assist the blended MODEL loses to the blended BASELINE (-2.1%).
# No tree beats the fitted F1-blend for the attacking stats (goals/assists) — the odds already price
# them and the F1 anchor does the work. Confirms score1/assist stay on the blend. Left gated.
_BLEND = os.environ.get("BLEND_FEATURE") == "1"
FEATURES = (["persistence", "momentum"]
            + [f"form{w}" for w in WINDOWS] + [f"form_n{w}" for w in WINDOWS]
            + [f"form_win{w}" for w in WINDOWS] + [f"form_loss{w}" for w in WINDOWS]
            + (["win_hat", "opp_hat"] if _WINHAT else [])   # win_pred outputs (env WINHAT_FEATURES=1)
            + (["blend"] if _BLEND else [])                 # fitted F1-blend (env BLEND_FEATURE=1)
            # real on-pitch trailing form + minutes (add_real_form) are OFF — tested 2026-08, added
            # nothing (the odds already price in real form + rotation; see memory). Re-enable by
            # un-commenting here AND the add_real_form call in main().
            # + [f"real_form{w}" for w in REAL_WINDOWS] + [f"real_n{w}" for w in REAL_WINDOWS]
            # + [f"rmin{w}" for w in REAL_WINDOWS]
            + ([f"xg_form{w}" for w in XG_WINDOWS] + [f"xa_form{w}" for w in XG_WINDOWS] if _XG else [])
            + (["predicted"] if _PRED else [])   # baseline-as-feature (env PRED_FEATURE=1)
            # peer_avg / rel_peer (add_peer_form) are OFF for now — re-add here + call it in main() to enable
            + ["own_title", "own_releg", "own_top6", "opp_title", "opp_releg", "opp_top6",
               "title_diff", "releg_diff", "top6_diff", "strength_diff",
               "position", "venue", "horizon"])
KEYS = ["season", "player_id", "player", "gw_from", "gw_to"]

# per-GW real outcome (binary) that mirrors each odds-market component, from player_gameweek_stats
REAL_OUTCOME = {
    "score1": ("goals_scored", 1), "assist": ("assists", 1), "yellow": ("yellow_cards", 1),
    "clean_sheet": ("clean_sheets", 1), "concede2": ("goals_conceded", 2), "saves3": ("saves", 3),
}


def _dense_trailing(archive, col, windows=WINDOWS, key="Player Name"):
    """Per (season, `key`): trailing mean and slot-count of `col` over the EXACT prior gameweek
    slots (M-1, M-2, ...) for each window. Gameweeks are densely reindexed so a missing week is a
    genuine empty slot, never back-filled from further back. Season-grouped, so form does NOT carry
    across seasons. Returns a frame indexed (Season, `key`, Gameweek) with mean{w}/count{w}."""
    a = archive[["Season", key, "Gameweek", col]].copy()
    a["Gameweek"] = pd.to_numeric(a["Gameweek"], errors="coerce")
    a[col] = pd.to_numeric(a[col], errors="coerce")
    a = a.dropna(subset=["Gameweek"])
    recs = []
    for (s, p), g in a.groupby(["Season", key]):
        ser = g.set_index("Gameweek")[col].sort_index()
        ser = ser[~ser.index.duplicated()]
        full = ser.reindex(range(int(ser.index.min()), int(ser.index.max()) + 1))  # gaps -> NaN slots
        prior = full.shift(1)
        rec = {"Season": s, key: p, "Gameweek": full.index}
        for w in windows:
            rec[f"mean{w}"] = prior.rolling(w, min_periods=1).mean().values
            rec[f"count{w}"] = prior.rolling(w, min_periods=1).count().values
        recs.append(pd.DataFrame(rec))
    return pd.concat(recs).set_index(["Season", key, "Gameweek"])


def load_real_gw():
    """Real per-gameweek on-pitch outcomes from the FPL-Core-Insights By-Gameweek dumps, keyed by
    FPL id + gameweek. Each folder's player_gameweek_stats.csv holds that single GW's per-player
    stats (minutes ~90, not season cumulative). Returns [Season, id, Gameweek, minutes, real_<stat>]
    where real_<stat> is the binary on-pitch version of each odds market (e.g. goals_scored>=1)."""
    pat = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", "*",
                       "By Gameweek", "GW*", "player_gameweek_stats.csv")
    frames = []
    for p in glob.glob(pat):
        d = pd.read_csv(p)
        if "gw" not in d.columns or "id" not in d.columns:
            continue
        g = pd.DataFrame({
            "Season": Path(p).parents[2].name,
            "id": pd.to_numeric(d["id"], errors="coerce").astype(float),
            "Gameweek": pd.to_numeric(d["gw"], errors="coerce"),
            "minutes": pd.to_numeric(d["minutes"], errors="coerce"),
            "xg": pd.to_numeric(d.get("expected_goals"), errors="coerce"),    # continuous per-GW xG
            "xa": pd.to_numeric(d.get("expected_assists"), errors="coerce"),  # continuous per-GW xA
        })
        for stat, (col, thr) in REAL_OUTCOME.items():
            g[f"real_{stat}"] = (pd.to_numeric(d[col], errors="coerce") >= thr).astype(float)
        frames.append(g)
    real = pd.concat(frames, ignore_index=True)
    return real.dropna(subset=["id", "Gameweek"]).drop_duplicates(subset=["Season", "id", "Gameweek"])


def add_real_form(pairs, real, windows=REAL_WINDOWS):
    """Trailing REAL performance (distinct from the odds-derived form{w}, which is the market's
    weekly expectation): what the player ACTUALLY did over the exact prior w gameweeks.
      real_form{w} = trailing mean of the component's real on-pitch outcome (stat-specific)
      real_n{w}    = how many of those w slots had real data (By-Gameweek dumps start ~GW10)
      rmin{w}      = trailing mean minutes — the objective rotation signal (a filter on actual
                     starters would condition on a FUTURE outcome; a trailing feature does not).
    Joined by (season, player_id, gw_from); strictly trailing, so leak-free."""
    pairs = pairs.copy()
    pid = pd.to_numeric(pairs["player_id"], errors="coerce").astype(float)
    idx_all = list(zip(pairs["season"], pid, pairs["gw_from"]))
    lut_min = _dense_trailing(real, "minutes", windows, key="id")
    for w in windows:
        pairs[f"rmin{w}"] = lut_min[f"mean{w}"].reindex(idx_all).values
        pairs[f"real_n{w}"] = lut_min[f"count{w}"].reindex(idx_all).values
    for w in windows:
        pairs[f"real_form{w}"] = float("nan")
    for stat in STATS:
        lut = _dense_trailing(real, f"real_{stat}", windows, key="id")
        m = pairs["stat"] == stat
        idx = list(zip(pairs.loc[m, "season"], pid[m], pairs.loc[m, "gw_from"]))
        for w in windows:
            pairs.loc[m, f"real_form{w}"] = lut[f"mean{w}"].reindex(idx).values
    return pairs


def add_xg_form(pairs, real, windows=XG_WINDOWS):
    """Trailing EXPECTED goals/assists over the exact prior w gameweeks — continuous and smoother
    than the binary real_form outcomes (players regress toward xG/xA, so it forecasts future goals/
    assists better than raw past goals/assists). Player-level (same value regardless of stat row);
    the per-component model uses xg_form for goals, xa_form for assists. Strictly trailing -> leak-free."""
    pairs = pairs.copy()
    pid = pd.to_numeric(pairs["player_id"], errors="coerce").astype(float)
    idx = list(zip(pairs["season"], pid, pairs["gw_from"]))
    for col, feat in (("xg", "xg_form"), ("xa", "xa_form")):
        lut = _dense_trailing(real, col, windows, key="id")
        for w in windows:
            pairs[f"{feat}{w}"] = lut[f"mean{w}"].reindex(idx).values
    return pairs


def add_form(pairs, archive, windows=WINDOWS):
    """Trailing form + the difficulty of the fixtures it was earned against, over each window w:
      form{w}   = mean of the player's own F1 <stat> over the exact prior w gameweeks (NaN only when
                  none were captured); form_n{w} = how many of those w slots had data
      form_win{w}/form_loss{w} = the team's mean F1 win/opponent-win over the same w slots, so form
                  can be read RELATIVE to how easy/hard those fixtures were. Gaps never skipped."""
    pairs = pairs.copy()
    idx_all = list(zip(pairs["season"], pairs["player"], pairs["gw_from"]))
    for w in windows:
        pairs[f"form{w}"] = float("nan")
        pairs[f"form_n{w}"] = float("nan")
    for stat, col in STATS.items():                              # stat-specific scoring form
        lut = _dense_trailing(archive, col, windows)
        m = pairs["stat"] == stat
        idx = list(zip(pairs.loc[m, "season"], pairs.loc[m, "player"], pairs.loc[m, "gw_from"]))
        for w in windows:
            pairs.loc[m, f"form{w}"] = lut[f"mean{w}"].reindex(idx).values
            pairs.loc[m, f"form_n{w}"] = lut[f"count{w}"].reindex(idx).values
    for col, base in (("F1 Win", "form_win"), ("F1 Opponent Win", "form_loss")):  # fixture difficulty
        lut = _dense_trailing(archive, col, windows)
        for w in windows:
            pairs[f"{base}{w}"] = lut[f"mean{w}"].reindex(idx_all).values
    return pairs


def add_peer_form(pairs, archive, windows=WINDOWS):
    """OBJECTIVE difficulty-adjustment (replaces the model-derived factor): the player's F1 <stat>
    relative to his same-position teammates that match. Fixture difficulty hits the whole positional
    group equally, so a player's value ABOVE his peers strips out difficulty with no model at all.
    Peers = the OTHER same-position teammates who are PRICED that week (a non-null value = the market
    expects them to feature — no reliance on our curated F1 Start). Self excluded; plain mean:
      peer_avg    = mean F1 <stat> of the player's priced same-position teammates at M
      rel_peer    = persistence - peer_avg (his standing above peers this fixture)
      rel_peer{w} = trailing mean of rel_peer over the exact prior w gameweeks (difficulty-adj. form).
    NaN when he has no priced positional peer (a genuine lone specialist — peer_avg flags it)."""
    pairs = pairs.copy()
    for f in ["peer_avg", "rel_peer"] + [f"rel_peer{w}" for w in windows]:
        pairs[f] = float("nan")
    keys = ["Season", "Gameweek", "Team", "Position"]
    for stat, col in STATS.items():
        val = pd.to_numeric(archive[col], errors="coerce")
        a = archive[["Season", "Player Name", "Gameweek", "Team", "Position"]].copy()
        a["val"] = val
        s = a.groupby(keys)["val"].transform("sum")                 # sum skips NaN (unpriced) players
        c = a.groupby(keys)["val"].transform("count")               # count of priced players
        denom = c - a["val"].notna().astype(int)                    # other priced same-position peers
        peer = (s - a["val"].fillna(0.0)) / denom.where(denom > 0)  # plain mean of OTHER priced peers
        rel = val - peer
        mi = pd.MultiIndex.from_arrays([a["Season"], a["Player Name"], a["Gameweek"]])
        peer_lut, rel_lut = pd.Series(peer.values, index=mi), pd.Series(rel.values, index=mi)
        m = pairs["stat"] == stat
        idx = list(zip(pairs.loc[m, "season"], pairs.loc[m, "player"], pairs.loc[m, "gw_from"]))
        pairs.loc[m, "peer_avg"] = peer_lut.reindex(idx).values
        pairs.loc[m, "rel_peer"] = rel_lut.reindex(idx).values
        tmp = archive[["Season", "Player Name", "Gameweek"]].copy()
        tmp["_rp"] = rel.values
        tl = _dense_trailing(tmp, "_rp", windows)
        for w in windows:
            pairs.loc[m, f"rel_peer{w}"] = tl[f"mean{w}"].reindex(idx).values
    return pairs


def add_diffs(pairs):
    """Relative-strength features (odds are probabilities): the model no longer has to learn the
    diffs from the raw own/opp pair. strength_diff mirrors win_pred's own sd = title+top6-releg."""
    pairs = pairs.copy()
    pairs["title_diff"] = pairs["own_title"] - pairs["opp_title"]
    pairs["releg_diff"] = pairs["own_releg"] - pairs["opp_releg"]
    pairs["top6_diff"] = pairs["own_top6"] - pairs["opp_top6"]
    pairs["strength_diff"] = ((pairs["own_title"] + pairs["own_top6"] - pairs["own_releg"])
                              - (pairs["opp_title"] + pairs["opp_top6"] - pairs["opp_releg"]))
    pairs["momentum"] = pairs["persistence"] - pairs["form3"]     # now vs recent form (trend)
    return pairs


def _mae(a, b):
    d = (a - b).abs()
    return float(d.mean()) if d.notna().any() else float("nan")


def main():
    archive = load_archive()
    pairs = build_pairs(archive)
    pairs = add_form(pairs, archive)
    # pairs = add_real_form(pairs, load_real_gw())   # OFF — real form/minutes added nothing (see FEATURES)
    if _XG:                                          # env XG_FEATURES=1: test trailing xG/xA
        pairs = add_xg_form(pairs, load_real_gw())
    pairs = add_diffs(pairs)
    pairs = pairs[pairs["valid_opp"]]                     # need the opponent's odds to have been known
    outdir = os.path.join(config.OUTPUTS_DIR, "training")
    os.makedirs(outdir, exist_ok=True)

    print(f"{'component':<12}{'rows':>7}{'players':>9}{'gw_pairs':>9} | "
          f"{'MAE base':>9}{'MAE persist':>12}  (baseline to beat / naive)")
    print("-" * 74)
    for stat in STATS:
        sub = pairs[(pairs["stat"] == stat)].dropna(subset=["actual", "persistence"]).copy()
        if _BLEND:                                          # w*predicted + (1-w)*persistence, the deployed F1-blend
            w = config.PROJECTION_BLEND.get(stat, 1.0)
            sub["blend"] = (w * pd.to_numeric(sub["predicted"], errors="coerce")
                            + (1 - w) * pd.to_numeric(sub["persistence"], errors="coerce"))
        extra = [c for c in ["predicted", "actual"] if c not in FEATURES]   # avoid dup when predicted is a feature
        out = sub[KEYS + FEATURES + extra]
        path = os.path.join(outdir, f"train_{stat}.csv")
        out.to_csv(path, index=False)
        print(f"{stat:<12}{len(out):>7}{out['player'].nunique():>9}"
              f"{out.groupby(['season', 'gw_from', 'horizon']).ngroups:>9} | "
              f"{_mae(sub['predicted'], sub['actual']):>9.4f}{_mae(sub['persistence'], sub['actual']):>12.4f}")
    print(f"\nwritten to {os.path.relpath(outdir, ROOT)}/train_<stat>.csv "
          f"({len(FEATURES)} features + predicted (baseline) + actual (target) per row)")


if __name__ == "__main__":
    main()
