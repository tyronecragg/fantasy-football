# -*- coding: utf-8 -*-
"""How good is the bonus model? Score the live odds-anchored bonus (model.bonus_points_odds) and the
old flat uplift against ACTUAL bonus points, and try a rank-aware alternative.

    env/Scripts/python tools/bonus_check.py

Rows: every archived 2025-26 gameweek (F1 = real odds), players with a curated start probability
> 0, joined to the bonus points they actually got (FPL-Core-Insights By-Gameweek). The live model
splits each team's bonus pot in PROPORTION to XP above the appearance floor; but real bonus is
3/2/1 to the top three BPS in the MATCH, i.e. winner-take-most. The alternative here keeps the pot
and the weights but shares them by a POWER of the weight (sharpness gamma, 1 = the live model, higher
= more to the best player), with gamma chosen leave-one-gameweek-out.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, model  # noqa: E402

SEASON = "2025-2026"


def load_actual():
    pat = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", SEASON, "By Gameweek", "GW*",
                       "player_gameweek_stats.csv")
    d = pd.concat([pd.read_csv(p, usecols=["id", "gw", "bonus", "minutes", "total_points"]) for p in glob.glob(pat)])
    return d.drop_duplicates(["id", "gw"]).rename(columns={"id": "player_id", "gw": "Gameweek"})


def share_power(weight, team, gamma):
    w = weight.clip(lower=0.0) ** gamma
    denom = w.groupby(team).transform("sum")
    return (w / denom).where(denom > 0, 0.0)


def main():
    a = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False)
    a = a[a["Season"] == SEASON].copy()
    for c in ["Gameweek", "player_id", "F1 XP Pre", "F1 Start", "F1 Win", "F1 Opponent Win", "F1 XP"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["Gameweek", "player_id", "F1 XP Pre", "F1 Win"])
    act = load_actual()
    d = a.merge(act, on=["player_id", "Gameweek"], how="left")
    d["bonus"] = d["bonus"].fillna(0.0)
    d = d[d["F1 Start"] > 0].copy()

    # live model, recomputed from the archived F1 inputs (the archive's own Bonus column is the OLD flat model)
    d["weight"] = (d["F1 XP Pre"] - 2.0 * d["F1 Start"]).clip(lower=0.0)
    d["weight_pos"] = d["weight"] * d["Position"].map(config.BONUS_POSITION_WEIGHT).fillna(0.0)
    d["key"] = d["Gameweek"].astype(int).astype(str) + ":" + d["Team"]
    gk_flat = (config.BONUS_GK_PER_START * d["F1 Start"]).where(d["Position"] == "GK", 0.0)
    d["live"] = model.bonus_points_odds(d["weight_pos"], d["key"], d["F1 Win"], d["F1 Opponent Win"], reserve=gk_flat) + gk_flat   # deployed
    d["unweighted"] = model.bonus_points_odds(d["weight"], d["key"], d["F1 Win"], d["F1 Opponent Win"])  # before multipliers
    b = model.COEFS["bonus"]
    d["flat"] = 2.0 * (b["intercept"] + b["slope"] * d["F1 XP Pre"]).clip(0, 1)
    pw = d["F1 Win"].fillna(0).clip(0, 1)
    pl = d["F1 Opponent Win"].fillna(0).clip(0, 1)
    d["pot"] = pw * model.BONUS_BY_RESULT["win"] + (1 - pw - pl).clip(0, 1) * model.BONUS_BY_RESULT["draw"] + pl * model.BONUS_BY_RESULT["lose"]

    def mae(x):
        return float((d[x] - d["bonus"]).abs().mean())

    print(f"{len(d)} player-weeks, {d['Gameweek'].nunique()} gameweeks; actual bonus per player-week {d['bonus'].mean():.3f}")
    print(f"\ntotal bonus allocated vs actual:  live {d['live'].sum() / d['bonus'].sum():.2f}x   flat {d['flat'].sum() / d['bonus'].sum():.2f}x")
    print(f"MAE vs actual bonus:  live {mae('live'):.4f}   flat {mae('flat'):.4f}   zero {float(d['bonus'].abs().mean()):.4f}")

    # calibration by expected-bonus band (live model)
    d["band"] = pd.cut(d["live"], [-0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0])
    print("\nlive model calibration by expected-bonus band:")
    print(d.groupby("band", observed=True).agg(n=("bonus", "size"), expected=("live", "mean"), actual=("bonus", "mean")).round(3).to_string())
    print("\nby position (deployed multipliers vs none; the archive sample holds ~79% of bonus winners, so ~1.2x totals are coverage):")
    print(d.groupby("Position").agg(n=("bonus", "size"), deployed=("live", "mean"), unweighted=("unweighted", "mean"), actual=("bonus", "mean")).round(3).to_string())

    # rank-aware alternative: share the pot by weight ** gamma, gamma chosen leave-one-GW-out
    gws = sorted(d["Gameweek"].unique())
    grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print("\nsharpness gamma (1.0 = live model): pooled MAE, and leave-one-gameweek-out choice")
    scores = {}
    for g in grid:
        d[f"g{g}"] = d["pot"] * share_power(d["weight"], d["key"], g)
        scores[g] = mae(f"g{g}")
        print(f"  gamma {g:<4} MAE {scores[g]:.4f}   top-band actual/expected: "
              f"{d[d[f'g{g}'] > 1.0]['bonus'].mean():.2f}/{d[d[f'g{g}'] > 1.0][f'g{g}'].mean():.2f} (n={int((d[f'g{g}'] > 1.0).sum())})")
    loo = []
    for gw in gws:
        tr = d[d["Gameweek"] != gw]
        best = min(grid, key=lambda g: float((tr[f"g{g}"] - tr["bonus"]).abs().mean()))
        te = d[d["Gameweek"] == gw]
        loo.append(float((te[f"g{best}"] - te["bonus"]).abs().mean()))
        scores.setdefault("loo_best", []).append(best)
    print(f"  leave-one-GW-out: chosen gammas {scores['loo_best']}, MAE {np.mean(loo):.4f} (live {mae('live'):.4f})")
    refine(d)


def refine(d):
    """Squared-error scoring (MAE rewards predicting zero for a sparse target), plus two cheap fixes:
    a per-position multiplier on the bonus weight (BPS rewards saves/clean-sheet minutes that XP
    under-weights for keepers) and a sharper within-team split. Multipliers are fitted so each
    position's expected total matches its actual total on the OTHER gameweeks (leave-one-GW-out)."""
    def rmse(x):
        return float(np.sqrt(((d[x] - d["bonus"]) ** 2).mean()))
    print(f"\nRMSE vs actual bonus:  zero {float(np.sqrt((d['bonus'] ** 2).mean())):.4f}   flat {rmse('flat'):.4f}   live {rmse('live'):.4f}")
    gws = sorted(d["Gameweek"].unique())
    for gamma in (1.0, 2.0, 3.0):
        col = f"pos_g{gamma}"
        d[col] = np.nan
        mults = []
        for gw in gws:
            tr, te = d["Gameweek"] != gw, d["Gameweek"] == gw
            mult = {p: 1.0 for p in d["Position"].unique()}
            for _ in range(8):                                  # fixed-point: scale each position to its actual total
                w = d["weight"] * d["Position"].map(mult)
                exp = d["pot"] * share_power(w, d["key"], gamma)
                for p in mult:
                    m = tr & (d["Position"] == p)
                    if exp[m].sum() > 0:
                        mult[p] *= float(d.loc[m, "bonus"].sum() / exp[m].sum()) ** 0.5
            w = d["weight"] * d["Position"].map(mult)
            d.loc[te, col] = (d["pot"] * share_power(w, d["key"], gamma))[te]
            mults.append(mult)
        avg = {p: round(float(np.mean([m[p] for m in mults])), 2) for p in mults[0]}
        print(f"  live + position multipliers (gamma {gamma}): RMSE {rmse(col):.4f}   multipliers {avg}   "
              f"total {d[col].sum() / d['bonus'].sum():.2f}x")
        if gamma == 1.0:
            print("    by position:", d.groupby("Position").agg(expected=(col, "mean"), actual=("bonus", "mean")).round(3).T.to_string())
    # calibration of the best variant by band
    best = "pos_g2.0"
    d["band2"] = pd.cut(d[best], [-0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 3.0])
    print(f"\n{best} calibration by band:")
    print(d.groupby("band2", observed=True).agg(n=("bonus", "size"), expected=(best, "mean"), actual=("bonus", "mean")).round(3).to_string())




if __name__ == "__main__":
    main()
