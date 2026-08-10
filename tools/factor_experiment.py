"""Factor-stabilisation experiment: which way of computing a player's factor best
predicts his future odds-implied probabilities?

Variants evaluated on the backtest harness (train = forecasts made GW16-24,
holdout = forecasts made GW25+), for score1 and assist:

  A. current   — single-week factor: this week's odds / baseline (live behaviour)
  B. median    — trailing median of weekly factors up to and including this week
  C. xg-blend  — w x A + (1-w) x xG-implied factor (from cumulative xG/xA per-90
                 snapshots as of the forecast week — no lookahead), with a global
                 per-stat calibration constant so the xG factor sits on the odds
                 convention scale; w fitted on train, judged on holdout
  D. median+xg — w x B + (1-w) x xG factor

Usage: python tools/factor_experiment.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model  # noqa: E402
from fpl_pipeline.names import apply_player_names  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "backtest_projections", os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_projections.py"))
bt = importlib.util.module_from_spec(spec)
sys.modules.setdefault("backtest_projections", bt)
spec.loader.exec_module(bt)

TRAIN_MAX_GW = 24          # forecasts made up to here fit w; later weeks judge
MIN_XG_MINUTES = 450       # need 5+ full matches of xG evidence
STATS = {"score1": ("F1 Score 1+", "expected_goals"),
         "assist": ("F1 Assist", "expected_assists")}


def weekly_factors(archive):
    """(player, gw) -> single-week odds factor, per stat — the raw material."""
    win, opp = archive["F1 Win"], archive["F1 Opponent Win"]
    pos, home = archive["Position"], archive["F1 Venue"] == "H"
    out = {}
    for stat, (prob_col, _) in STATS.items():
        f = archive[prob_col] / model.baseline(stat, win, opp, pos, home)
        out[stat] = pd.DataFrame({"player": archive["Player Name"],
                                  "gw": archive["Gameweek"], "factor": f}).dropna()
    return out


def xg_snapshots():
    """(player, gw) -> cumulative xG90/xA90 as of that gameweek (2025-26 data)."""
    base = os.path.join(config.ROOT, "fpl_data", "FPL-Core-Insights", "data", "2025-2026")
    players = pd.read_csv(os.path.join(base, "players.csv"))
    stats = pd.read_csv(os.path.join(base, "playerstats.csv"))
    players["name"] = apply_player_names(players["first_name"] + " " + players["second_name"])
    df = stats.merge(players[["player_id", "name"]], left_on="id", right_on="player_id")
    df = df[df["minutes"] >= MIN_XG_MINUTES]
    for col in ("expected_goals", "expected_assists"):
        df[f"{col}_90"] = pd.to_numeric(df[col], errors="coerce") / df["minutes"] * 90
    return df[["name", "gw", "expected_goals_90", "expected_assists_90"]]


def neutral_baseline(stat, positions):
    """Fixture-average baseline per position (league-average odds, home/away mean)."""
    n = len(positions)
    avg = pd.Series(0.37, index=range(n))
    pos = pd.Series(list(positions), index=range(n))
    home = model.baseline(stat, avg, avg, pos, pd.Series(True, index=range(n)))
    away = model.baseline(stat, avg, avg, pos, pd.Series(False, index=range(n)))
    return dict(zip(positions, (home + away) / 2))


def build_variant_tables(archive):
    """Per stat: dicts (player, gw) -> factor for each variant (A implicit)."""
    raw = weekly_factors(archive)
    xg = xg_snapshots()
    pos_of = archive.drop_duplicates("Player Name").set_index("Player Name")["Position"]
    tables = {}
    for stat, (_, xg_col) in STATS.items():
        f = raw[stat].sort_values("gw")

        # B: trailing median (includes current week; strictly no lookahead)
        f["median"] = f.groupby("player")["factor"].transform(
            lambda s: s.expanding().median())
        median = {(r.player, r.gw): r.median for r in f.itertuples()}

        # xG factor: latest cumulative snapshot with gw <= forecast week
        nb = neutral_baseline(stat, ["GK", "DEF", "MID", "FWD"])
        xg_sorted = xg.sort_values("gw")
        xg_lookup = {}
        for name, grp in xg_sorted.groupby("name"):
            gws, vals = grp["gw"].values, grp[f"{xg_col}_90"].values
            xg_lookup[name] = (gws, vals)

        def xg_factor(player, week):
            if player not in xg_lookup or player not in pos_of.index:
                return np.nan
            gws, vals = xg_lookup[player]
            idx = np.searchsorted(gws, week, side="right") - 1
            if idx < 0 or not np.isfinite(vals[idx]):
                return np.nan
            p = 1.0 - np.exp(-vals[idx])
            return p / nb[str(pos_of[player])]

        xgf = {(r.player, r.gw): xg_factor(r.player, r.gw) for r in f.itertuples()}
        single = {(r.player, r.gw): r.factor for r in f.itertuples()}
        tables[stat] = {"single": single, "median": median, "xg": xgf}
    return tables


def evaluate(archive, tables):
    def mae(pairs, split):
        sub = pairs[pairs["valid_opp"]]
        sub = sub[sub["gw_from"] <= TRAIN_MAX_GW] if split == "train" else sub[sub["gw_from"] > TRAIN_MAX_GW]
        return {stat: (s["predicted"] - s["actual"]).abs().mean()
                for stat, s in sub.dropna(subset=["predicted", "actual"]).groupby("stat")}

    def blended(stat, base, w, calib):
        b, x = tables[stat][base], tables[stat]["xg"]
        return {k: (w * b[k] + (1 - w) * calib * x[k]) if np.isfinite(x.get(k, np.nan)) else b[k]
                for k in b}

    # calibration: put the xG factor on the odds-convention scale (train weeks only)
    calib = {}
    for stat in STATS:
        train_keys = [k for k in tables[stat]["single"] if k[1] <= TRAIN_MAX_GW]
        pairs_both = [(tables[stat]["single"][k], tables[stat]["xg"].get(k, np.nan)) for k in train_keys]
        odds_vals = np.array([a for a, b in pairs_both if np.isfinite(b)])
        xg_vals = np.array([b for a, b in pairs_both if np.isfinite(b)])
        calib[stat] = odds_vals.mean() / xg_vals.mean()
        print(f"{stat}: xG calibration constant = {calib[stat]:.3f} "
              f"({len(xg_vals)} player-weeks with xG)")

    results = {}
    variants = {"A_current": None,
                "B_median": {s: tables[s]["median"] for s in STATS}}
    for name, ov in variants.items():
        pairs = bt.build_pairs(archive, factor_overrides=ov)
        results[name] = {"train": mae(pairs, "train"), "holdout": mae(pairs, "holdout")}

    # C/D: fit w per stat on train, evaluate at the chosen w on holdout
    for label, base in (("C_xg_single", "single"), ("D_xg_median", "median")):
        best_w = {}
        for w in np.arange(0.0, 1.01, 0.1):
            ov = {s: blended(s, base, w, calib[s]) for s in STATS}
            tr = mae(bt.build_pairs(archive, factor_overrides=ov), "train")
            for s in STATS:
                if s not in best_w or tr[s] < best_w[s][1]:
                    best_w[s] = (w, tr[s])
        ov = {s: blended(s, base, best_w[s][0], calib[s]) for s in STATS}
        pairs = bt.build_pairs(archive, factor_overrides=ov)
        results[label] = {"train": mae(pairs, "train"), "holdout": mae(pairs, "holdout"),
                          "w": {s: best_w[s][0] for s in STATS}}
    return results


def main():
    archive = bt.load_archive()
    archive = archive[archive["Season"] == "2025-2026"]
    tables = build_variant_tables(archive)
    results = evaluate(archive, tables)

    print(f"\n{'variant':<14} {'stat':<8} {'train MAE':>10} {'HOLDOUT MAE':>12}   w")
    for name, r in results.items():
        for stat in STATS:
            w = r.get("w", {}).get(stat, "")
            print(f"{name:<14} {stat:<8} {r['train'][stat]:>10.4f} {r['holdout'][stat]:>12.4f}   {w}")
    return results


if __name__ == "__main__":
    main()
