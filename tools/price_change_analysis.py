"""How do FPL prices move after GW1, by initial ownership?

    python tools/price_change_analysis.py [--season 2025-2026]

Reads the per-gameweek FPL snapshots and tracks every player who existed at GW1,
bucketing by their GW1 `selected_by_percent` and reporting how their price moved.

Why ownership is the interesting axis: FPL price changes are driven by NET TRANSFERS,
and ownership sets the pool on both sides. A 45%-owned player has a huge base of
potential sellers (crash risk) but a shrinking pool of potential buyers; a 2%-owned
player has almost unlimited room to rise but needs a reason. The question this answers
is which effect dominates in practice, and how early it resolves.

Prices matter to us through sell value, not scoreline: a rise only returns half its
value on exit (see fpl_pipeline/prices.py), while a fall costs full price.
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

BUCKETS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 100)]


def load_season(season):
    """{gw: DataFrame} of per-gameweek player snapshots."""
    root = os.path.join(config.ROOT, "fpl_data", "FPL-Core-Insights", "data", season,
                        "By Gameweek")
    frames = {}
    for path in sorted(glob.glob(os.path.join(root, "GW*", "playerstats.csv"))):
        gw = int(os.path.basename(os.path.dirname(path))[2:])
        d = pd.read_csv(path, usecols=lambda c: c in {
            "id", "web_name", "now_cost", "selected_by_percent", "total_points"})
        frames[gw] = d.drop_duplicates(subset="id").set_index("id")
    return frames


def build(frames):
    gw1 = frames[1]
    out = pd.DataFrame({
        "web_name": gw1["web_name"],
        "own1": gw1["selected_by_percent"],
        "price1": gw1["now_cost"],
    })
    for gw in sorted(frames):
        if gw > 1:
            out[f"price{gw}"] = frames[gw]["now_cost"]
    last = max(frames)
    out["pts_gw1_6"] = frames[min(6, last)]["total_points"]
    return out.dropna(subset=["own1", "price1"]), last


def report(df, last_gw):
    checkpoints = [gw for gw in (2, 3, 5, 9, 19, last_gw) if f"price{gw}" in df.columns]

    print("=" * 92)
    print("PRICE CHANGE AFTER GW1, BY INITIAL OWNERSHIP")
    print("=" * 92)
    header = f"{'GW1 ownership':<16}{'n':>5}" + "".join(f"{'by GW' + str(g):>11}" for g in checkpoints)
    print(header)
    print("-" * len(header))
    for lo, hi in BUCKETS:
        sub = df[(df["own1"] >= lo) & (df["own1"] < hi)]
        if sub.empty:
            continue
        row = f"{f'{lo}-{hi}%':<16}{len(sub):>5}"
        for gw in checkpoints:
            row += f"{(sub[f'price{gw}'] - sub['price1']).mean():>+11.3f}"
        print(row)

    print(f"\nShare of each bucket that ROSE / FELL by GW{checkpoints[-2] if len(checkpoints) > 1 else last_gw}"
          f" (a rise only returns half its value on sale; a fall costs full price)")
    gw = checkpoints[-2] if len(checkpoints) > 1 else last_gw
    print(f"{'GW1 ownership':<16}{'n':>5}{'rose':>9}{'fell':>9}{'flat':>9}{'mean rise':>11}{'mean fall':>11}")
    for lo, hi in BUCKETS:
        sub = df[(df["own1"] >= lo) & (df["own1"] < hi)]
        if sub.empty:
            continue
        chg = sub[f"price{gw}"] - sub["price1"]
        rose, fell = chg > 0.001, chg < -0.001
        print(f"{f'{lo}-{hi}%':<16}{len(sub):>5}{rose.mean():>8.0%}{fell.mean():>9.0%}"
              f"{(~rose & ~fell).mean():>9.0%}"
              f"{(chg[rose].mean() if rose.any() else 0):>+11.2f}"
              f"{(chg[fell].mean() if fell.any() else 0):>+11.2f}")

    # Ownership vs performance: which actually drives the early price move?
    early = [gw for gw in (5, 9) if f"price{gw}" in df.columns]
    if early and "pts_gw1_6" in df.columns:
        gw = early[-1]
        d = df.dropna(subset=["pts_gw1_6"]).copy()
        d["chg"] = d[f"price{gw}"] - d["price1"]
        d["own_half"] = (d["own1"] >= d["own1"].median()).map({True: "high own", False: "low own"})
        d["pts_half"] = (d["pts_gw1_6"] >= d["pts_gw1_6"].median()).map({True: "scored well", False: "scored poorly"})
        print(f"\nMean price change by GW{gw}, ownership vs early points (which drives it?)")
        piv = d.pivot_table(index="own_half", columns="pts_half", values="chg", aggfunc="mean")
        print(piv.round(3).to_string())
        print(f"\ncorrelation with price change by GW{gw}:  "
              f"initial ownership {d['own1'].corr(d['chg']):+.3f}   "
              f"early points {d['pts_gw1_6'].corr(d['chg']):+.3f}")

    print("\nBiggest early risers and fallers (GW1 ownership in brackets):")
    gw = early[-1] if early else last_gw
    d = df.copy()
    d["chg"] = d[f"price{gw}"] - d["price1"]
    for label, rows in (("RISERS", d.nlargest(8, "chg")), ("FALLERS", d.nsmallest(8, "chg"))):
        print(f"  {label}: " + ", ".join(
            f"{r.web_name} {r.chg:+.1f} ({r.own1:.0f}%)" for r in rows.itertuples()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-2026")
    args = ap.parse_args()
    frames = load_season(args.season)
    df, last = build(frames)
    print(f"season {args.season}: {len(df)} players present at GW1, snapshots through GW{last}\n")
    report(df, last)
