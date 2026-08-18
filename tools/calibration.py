"""Are the pipeline's player probabilities calibrated, or are longshots overstated?

    python tools/calibration.py

THE QUESTION. config.MARGIN_PLAYER applies a flat 1.05 haircut to every player price, but
the total load on Betway's goalscorer market measures ~3.5x (tools/margin_goals.py) while
the favourite's price is already near fair. That means the load is concentrated on
longshots, and the pipeline should be OVERSTATING unlikely scorers relative to likely ones.
Three attempts to fit the margin shape from prices all failed (see that tool, and
fit_margin_power.py) because prices alone underdetermine it.

Outcomes do not. Bucket players by the probability the pipeline gave them, then compare
with how often it actually happened. No functional form, no distributional assumption —
if the 0-2% bucket scores 0.5% of the time, longshots are overstated by 4x and we can see
the shape directly.

DATA. Projections come from the archive (inputs/historical_player_data.csv: Season,
Gameweek, Player Name, "F1 Score 1+" = P(scores in that gameweek's fixture)). Outcomes come
from the FPL feed's per-match files, summed per player per gameweek. 2025-26 GW16-29.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

SEASON = "2025-2026"
MIN_MINUTES = 60   # see the control in main()


def actuals(season=SEASON):
    """goals scored per player per gameweek, from the per-match files."""
    base = os.path.join(os.path.dirname(config.FPL_DATA_DIR.rstrip("\/")), season)
    players = pd.read_csv(os.path.join(base, "players.csv"))
    name = {r.player_id: f"{r.first_name} {r.second_name}".strip()
            for r in players.itertuples()}
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "By Gameweek", "GW*")),
                    key=lambda p: int(re.search(r"GW(\d+)", p).group(1))):
        gw = int(re.search(r"GW(\d+)", d).group(1))
        try:
            m = pd.read_csv(os.path.join(d, "playermatchstats.csv"))
        except FileNotFoundError:
            continue
        col = "goals_scored" if "goals_scored" in m else "goals"
        if col not in m:
            continue
        mins = "minutes_played" if "minutes_played" in m else "minutes"
        g = m.groupby("player_id").agg(**{col: (col, "sum"),
                                          "minutes": (mins, "sum")}).reset_index()
        g["Gameweek"] = gw
        g["Player Name"] = g["player_id"].map(name)
        rows.append(g.rename(columns={col: "goals"}))
    return pd.concat(rows, ignore_index=True)


def main():
    arch = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"))
    arch = arch[arch["Season"] == SEASON][["Season", "Gameweek", "Player Name", "F1 Score 1+"]]
    arch = arch.dropna(subset=["F1 Score 1+"])
    act = actuals()
    df = arch.merge(act[["Gameweek", "Player Name", "goals", "minutes"]],
                    on=["Gameweek", "Player Name"], how="inner")
    df["scored"] = (df["goals"] > 0).astype(int)
    # CRITICAL CONTROL: a projected player who never came on cannot score, so an unfiltered
    # comparison blames the margin for what is really start probability. Restrict to players
    # who actually took the field.
    if MIN_MINUTES:
        before = len(df)
        df = df[df["minutes"] >= MIN_MINUTES]
        print(f"  filtered to minutes >= {MIN_MINUTES}: {before:,} -> {len(df):,} rows")
    print(f"{len(df):,} player-gameweeks joined "
          f"(GW{df.Gameweek.min()}-{df.Gameweek.max()}, {df['Player Name'].nunique()} players)\n")

    edges = [0, .02, .05, .08, .12, .18, .25, .35, .50, 1.01]
    print(f"{'projected P(score)':<22}{'n':>7}{'mean proj':>11}{'actual':>9}{'ratio':>8}")
    for lo, hi in zip(edges, edges[1:]):
        sub = df[(df["F1 Score 1+"] >= lo) & (df["F1 Score 1+"] < hi)]
        if len(sub) < 40:
            continue
        proj, real = sub["F1 Score 1+"].mean(), sub["scored"].mean()
        flag = "  <-- overstated" if real > 0 and proj / real > 1.3 else ""
        print(f"  {lo:.0%}-{hi:.0%}{'':<12}{len(sub):>7}{proj:>11.1%}{real:>9.1%}"
              f"{proj/max(real,1e-9):>8.2f}x{flag}")
    print(f"\noverall: mean projection {df['F1 Score 1+'].mean():.1%}  "
          f"actual {df['scored'].mean():.1%}  ratio {df['F1 Score 1+'].mean()/df['scored'].mean():.2f}x")
    print("\nratio > 1 = the pipeline says it happens more often than it does.")
    print("A ratio that RISES as the probability falls is the longshot bias we suspect.")


if __name__ == "__main__":
    main()


def fit_curve(df, col="F1 Score 1+"):
    """Monotone correction factor by projected probability, from TRAIN rows only.

    Returns knots (x = projected prob, y = multiplier). Interpolated in log space because
    the distortion grows multiplicatively as the price lengthens.
    """
    edges = [0, .05, .08, .12, .18, .25, .35, 1.01]
    xs, ys = [], []
    for lo, hi in zip(edges, edges[1:]):
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if len(sub) < 30:
            continue
        proj, real = sub[col].mean(), sub["scored"].mean()
        xs.append(proj)
        ys.append(min(1.0, real / proj) if proj > 0 else 1.0)   # never inflate
    # enforce monotone-nondecreasing multiplier as probability rises
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1])
    return np.array(xs), np.array(ys)


def apply_curve(p, xs, ys):
    return np.clip(p * np.interp(np.log(np.clip(p, 1e-6, 1)), np.log(xs), ys), 0, 1)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9); y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def validate(split=23):
    """Fit on early gameweeks, score on later ones. Fitting and scoring the same rows
    would flatter the correction for free."""
    arch = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"),
                       low_memory=False)
    arch = arch[arch["Season"] == SEASON][["Gameweek", "Player Name", "F1 Score 1+"]].dropna()
    act = actuals()
    df = arch.merge(act[["Gameweek", "Player Name", "goals", "minutes"]],
                    on=["Gameweek", "Player Name"], how="inner")
    df = df[df["minutes"] >= MIN_MINUTES].copy()
    df["scored"] = (df["goals"] > 0).astype(int)
    tr, te = df[df.Gameweek < split], df[df.Gameweek >= split]
    xs, ys = fit_curve(tr)
    print(f"\nOUT-OF-SAMPLE GATE  (fit GW<{split}: {len(tr)} rows | test GW>={split}: {len(te)} rows)")
    print("  correction knots: " + "  ".join(f"{x:.0%}->x{y:.2f}" for x, y in zip(xs, ys)))
    raw, cor = te["F1 Score 1+"].to_numpy(), apply_curve(te["F1 Score 1+"].to_numpy(), xs, ys)
    y = te["scored"].to_numpy()
    print(f"\n  {'':<12}{'Brier':>10}{'log loss':>11}{'mean pred':>11}{'actual':>9}")
    print(f"  {'current':<12}{brier(raw,y):>10.5f}{logloss(raw,y):>11.5f}"
          f"{raw.mean():>11.1%}{y.mean():>9.1%}")
    print(f"  {'corrected':<12}{brier(cor,y):>10.5f}{logloss(cor,y):>11.5f}"
          f"{cor.mean():>11.1%}{y.mean():>9.1%}")
    db, dl = brier(raw,y)-brier(cor,y), logloss(raw,y)-logloss(cor,y)
    print(f"  {'improvement':<12}{db:>+10.5f}{dl:>+11.5f}   "
          f"({db/brier(raw,y):+.1%} Brier, {dl/logloss(raw,y):+.1%} log loss)")
    print("\n  positive = corrected is better. Adopt only if BOTH improve.")


def horizon(max_f=8):
    """Does the longshot bias look the SAME at F1 as at F5, or does it change with horizon?

    This decides whether one correction can serve every fixture. Factors are
    probability/baseline and future fixtures are factor x baseline, so a correction applied
    at F1 rescales F2-F8 by the same multiplier. That is only right if the bias is the same
    size out there. It might not be: F1 comes from real market odds, while F2+ are model
    projections, and the bias is a property of the PRICE LEVEL rather than of the player.

    Archive row at gameweek G holds "Fk Score 1+" = the projection for gameweek G+k-1.
    """
    arch = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"),
                       low_memory=False)
    arch = arch[arch["Season"] == SEASON]
    act = actuals()
    act = act[act["minutes"] >= MIN_MINUTES]
    act["scored"] = (act["goals"] > 0).astype(int)

    print(f"\nBIAS BY HORIZON  (ratio = projected / actual; >1 means overstated)")
    print(f"{'':6}{'n':>7}{'mean proj':>11}{'actual':>9}{'overall':>9}"
          f"{'  <8%':>8}{'8-18%':>8}{'>18%':>8}")
    for k in range(1, max_f + 1):
        col = f"F{k} Score 1+"
        if col not in arch.columns:
            continue
        a = arch[["Gameweek", "Player Name", col]].dropna().copy()
        a["target_gw"] = a["Gameweek"] + (k - 1)
        m = a.merge(act[["Gameweek", "Player Name", "scored"]]
                    .rename(columns={"Gameweek": "target_gw"}),
                    on=["target_gw", "Player Name"], how="inner")
        if len(m) < 200:
            continue
        p, y = m[col], m["scored"]
        def band(lo, hi):
            s = m[(m[col] >= lo) & (m[col] < hi)]
            if len(s) < 40 or s["scored"].mean() == 0:
                return float("nan")
            return s[col].mean() / s["scored"].mean()
        print(f"  F{k}{'':<3}{len(m):>7}{p.mean():>11.1%}{y.mean():>9.1%}"
              f"{p.mean()/max(y.mean(),1e-9):>8.2f}x"
              f"{band(0,.08):>7.2f}x{band(.08,.18):>7.2f}x{band(.18,1.01):>7.2f}x")
    print("\n  If the per-band ratios hold roughly steady across F1..F8, ONE correction")
    print("  serves all horizons. If they drift, F2-F8 need their own treatment.")
