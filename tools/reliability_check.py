# -*- coding: utf-8 -*-
"""Measure the optimiser's RELIABILITY weights in POINTS space, and test candidate weight rows.

    env/Scripts/python tools/reliability_check.py

For every archived gameweek M and horizon k (F1..F8), take the XP the pipeline projected AT M for
fixture F{k} and the FPL points the player actually scored in gameweek M+k-1 (FPL-Core-Insights
By-Gameweek dumps). Then per horizon:
  skill_k   = 1 - MAE(F{k} XP, points) / MAE(positional-mean XP, points)   (the reliability yardstick)
  slope_k   = OLS slope of points on F{k} XP                                 (how much to trust a unit of XP)
  corr_k    = Spearman rank correlation                                      (does the ranking hold up?)
Reliability = skill_k / skill_1 (so F2_VS_F1 is MEASURED, not assumed).

Then the candidate rows: each is combined with the ownership weights into an 8-fixture weighted
XP at M, and scored by how well it ranks players on the points they actually went on to score over
those 8 gameweeks (Spearman, and points captured by the top-30 weighted-XP players). NOTE: this scores
the vectors as FORECASTS; the ownership part is about transfer flexibility, which no forecast test
can see, so the comparison is fairest between rows that share the same ownership component.
Caveat: the archive's XP columns are the projections as made then (2025-26, GW16-29, older models).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402

SEASON = "2025-2026"
OWN = [1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558]
ROWS = {
    "old":       [1.0, 0.850, 0.616, 0.578, 0.503, 0.498, 0.481, 0.472],
    "measured":  [1.0, 0.850, 0.732, 0.706, 0.704, 0.685, 0.664, 0.642],
    "proposed":  [1.0, 0.850, 0.710, 0.660, 0.640, 0.610, 0.570, 0.530],
    "points-measured": [1.0, 0.853, 0.714, 0.625, 0.546, 0.472, 0.40, 0.33],   # F7/F8 extrapolated
    "flat":      [1.0] * 8,
    "F1 only":   [1.0] + [0.0] * 7,
}


def load_points():
    pat = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", SEASON, "By Gameweek", "GW*",
                       "player_gameweek_stats.csv")
    frames = []
    for p in glob.glob(pat):
        d = pd.read_csv(p, usecols=["id", "gw", "total_points", "minutes"])
        frames.append(d)
    pts = pd.concat(frames).drop_duplicates(["id", "gw"])
    return pts.rename(columns={"id": "player_id", "gw": "Gameweek"})


def main():
    a = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False)
    a = a[a["Season"] == SEASON].copy()
    a["Gameweek"] = pd.to_numeric(a["Gameweek"], errors="coerce")
    a["player_id"] = pd.to_numeric(a["player_id"], errors="coerce")
    for k in range(1, 9):
        a[f"F{k} XP"] = pd.to_numeric(a[f"F{k} XP"], errors="coerce")
    pts = load_points()
    pts["player_id"] = pd.to_numeric(pts["player_id"], errors="coerce")
    ptab = pts.set_index(["player_id", "Gameweek"])["total_points"]

    # long frame: one row per (M, player, k) with the forecast and the realised points at M+k-1
    rows = []
    for k in range(1, 9):
        d = a[["Gameweek", "player_id", "Player Name", "Position", f"F{k} XP"]].dropna().copy()
        d["target_gw"] = d["Gameweek"] + k - 1
        d["points"] = ptab.reindex(list(zip(d["player_id"], d["target_gw"]))).values
        d = d.dropna(subset=["points"])
        d["k"] = k
        d = d.rename(columns={f"F{k} XP": "xp"})
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["pos_mean"] = L.groupby(["Gameweek", "k", "Position"])["xp"].transform("mean")

    print(f"{len(L)} forecast/outcome rows, gameweeks {int(L.Gameweek.min())}-{int(L.Gameweek.max())}, "
          f"targets through GW{int(L.target_gw.max())}\n")
    print(f"{'horizon':<8}{'n':>7}{'weeks':>6} | {'MAE xp':>7}{'MAE mean':>9}{'skill':>7} | {'slope':>6}{'spearman':>9} | {'rel=skill/F1':>13}")
    skill = {}
    have = [k for k in range(1, 9) if (L["k"] == k).any()]     # 2025-26 archived no F7/F8 XP (horizon was
    # F6 then); the 2026-27 archive stores them, so F7/F8 become measurable once outcomes exist -
    # the GW1 F8 forecast resolves at GW8, ~4-5 target weeks per horizon by GW12. Until then the
    # deployed row extrapolates the measured ~0.08/week decline.
    missing = [k for k in range(1, 9) if k not in have]
    for k in have:
        s = L[L["k"] == k]
        mae_x = (s["xp"] - s["points"]).abs().mean()
        mae_m = (s["pos_mean"] - s["points"]).abs().mean()
        sk = 1 - mae_x / mae_m
        slope = np.polyfit(s["xp"], s["points"], 1)[0]
        rho = spearmanr(s["xp"], s["points"]).correlation
        skill[k] = sk
        print(f"F{k:<7}{len(s):>7}{s['target_gw'].nunique():>6} | {mae_x:>7.3f}{mae_m:>9.3f}{sk:>7.3f} | "
              f"{slope:>6.3f}{rho:>9.3f} | {sk / skill[1]:>13.3f}")
    if missing:
        print(f"(no archived forecasts for F{missing[0]}..F{missing[-1]}: the 2025-26 archive predates the 8-fixture "
              f"horizon; measurable from the 2026-27 archive once ~GW12. Extrapolated in optimisation.RELIABILITY_WEIGHTS)")

    # candidate rows as 8-fixture forecasts: rank players at M on weighted XP vs points over the next 8 GWs
    print("\ncandidate rows x ownership as 8-fixture forecasts (players with all 8 forecasts and outcomes):")
    wide = L.pivot_table(index=["Gameweek", "player_id"], columns="k", values=["xp", "points"])
    wide = wide.dropna()
    X = wide["xp"][have].values
    P = wide["points"][have].values
    real8 = P.sum(axis=1)
    print(f"  {len(wide)} player-weeks over F1..F{max(have)}. Score = Spearman(weighted XP, realised points over")
    print(f"  those fixtures) and the share of the best-possible points captured by the top-30 players each week.")
    print(f"  {'row':<16}{'spearman':>9}{'top30 share':>12}   weights")
    for name, rel in ROWS.items():
        w = np.array([o * r for o, r in zip(OWN, rel)])[:len(have)]
        wxp = X @ w
        rho = spearmanr(wxp, real8).correlation
        shares = []
        for gw, idx in wide.groupby(level="Gameweek").indices.items():
            order = np.argsort(-wxp[idx])[:30]
            best = np.sort(real8[idx])[::-1][:30].sum()
            shares.append(real8[idx][order].sum() / best)
        print(f"  {name:<16}{rho:>9.3f}{np.mean(shares):>12.3f}   {[round(float(x), 2) for x in w]}")


if __name__ == "__main__":
    main()
