# -*- coding: utf-8 -*-
"""How fast should a player's OWN DefCon rate be trusted? Walk-forward test of the evidence gate.

    env/Scripts/python tools/dc_gate_backtest.py

Same as-of rebuild as tools/dc_backtest.py (truncated FPL dumps through the real ingest, 2025-26 GW16-29 +
2026-27 GW2-5), everything else at the LIVE settings (neutralised rate, full-90/subbed mixture, favourite
curve). Only the shrink toward the population average varies:
    capped G : weight = min(nineties / G, 1)      (live: G = config.DC_SHRINK_NINETIES)
    bayes k  : weight = nineties / (nineties + k)  (never reaches 1 - the textbook form)
`nineties` is true evidence (this season's minutes + capped prior-season minutes) / 90.
"""
import glob
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest, model, rating  # noqa: E402

DATA = os.path.dirname(config.FPL_DATA_DIR.rstrip("\\/"))
PRM = pd.read_csv(os.path.join(config.INPUTS_DIR, "dc_params.csv")).set_index("position")
ARCH = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False,
                   usecols=["Season", "Gameweek", "player_id", "F1 Win"])
for c in ["Gameweek", "player_id", "F1 Win"]:
    ARCH[c] = pd.to_numeric(ARCH[c], errors="coerce")
MATCHES = rating.load_matches([rating.season_code(s) for s in ["2025-2026", "2026-2027"]])
TMP = tempfile.mkdtemp(prefix="dc_gate_")
VARIANTS = [("capped", 1.0), ("capped", 2.0), ("capped", 3.0), ("capped", 4.0), ("capped", 6.0), ("capped", 8.0), ("capped", 12.0), ("capped", 16.0),
            ("bayes", 4.0), ("bayes", 8.0), ("bayes", 12.0), ("bayes", 20.0)]


def copy_season(season, upto=None, root=TMP):
    src, dst = os.path.join(DATA, season), os.path.join(root, season)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in ["players.csv", "teams.csv"]:
        shutil.copy(os.path.join(src, f), dst)
    ps = pd.read_csv(os.path.join(src, "playerstats.csv"), low_memory=False)
    if upto is not None:
        ps = ps[ps["gw"] < upto]
    ps.to_csv(os.path.join(dst, "playerstats.csv"), index=False)
    for p in glob.glob(os.path.join(src, "By Gameweek", "GW*", "player_gameweek_stats.csv")):
        gw = int(re.search(r"GW(\d+)", p).group(1))
        if upto is None or gw < upto:
            d = os.path.join(dst, "By Gameweek", f"GW{gw}")
            os.makedirs(d, exist_ok=True)
            shutil.copy(p, d)
    return dst


def cutoff_date(season, gw):
    m = pd.read_csv(os.path.join(DATA, season, "By Gameweek", f"GW{gw}", "matches.csv"))
    m = m[m["match_id"].astype(str).str.contains("-prem-")]
    col = [c for c in m.columns if "kickoff" in c.lower()][0]
    return pd.to_datetime(m[col], utc=True).min().tz_localize(None).normalize()


def names_by_id(season):
    pl = pd.read_csv(os.path.join(DATA, season, "players.csv"))
    nm = ingest._apply_name_changes(pl["first_name"] + " " + pl["second_name"])
    return dict(zip(pl["player_id"], nm)), dict(zip(pl["player_id"], pl["position"].map(config.POSITION_MAP)))


def main():
    rows = []
    targets = [("2025-2026", g) for g in range(16, 30)] + [("2026-2027", g) for g in range(2, 6)]
    solo = os.path.join(TMP, "solo"); os.makedirs(solo, exist_ok=True)
    copy_season("2025-2026")                                  # full prior season for the 2026-27 targets
    for season, gw in targets:
        act_p = os.path.join(DATA, season, "By Gameweek", f"GW{gw}", "player_gameweek_stats.csv")
        if not os.path.exists(act_p):
            continue
        act = pd.read_csv(act_p, usecols=["id", "minutes", "starts", "defensive_contribution"]).drop_duplicates("id")
        act = act[act["starts"] == 1]
        win = (ARCH[(ARCH.Season == season) & (ARCH.Gameweek == gw)].dropna(subset=["player_id"])
               .drop_duplicates("player_id").set_index("player_id")["F1 Win"])
        if win.dropna().empty:
            continue
        dst = copy_season(season, upto=gw, root=solo if season == "2025-2026" else TMP)
        config.SEASON, config.FPL_DATA_DIR = season, dst
        cut = cutoff_date(season, gw)
        ingest._load_favourite_mix = lambda seasons, _c=cut: ingest.favourite_mix(MATCHES[MATCHES["date"] < _c])
        id2name, id2pos = names_by_id(season)
        dc = ingest.load_defensive_contributions()             # live settings: neutralised dc90, nineties, p90
        for pos in ["DEF", "MID"]:
            tab = dc[pos].drop_duplicates("name").set_index("name")
            n = tab["nineties"].fillna(0.0).astype(float)
            avg = float(tab.loc[n >= 4.0, "dc90"].mean())      # population average held fixed across variants
            a = act[act["id"].map(id2pos) == pos].copy()
            a["name"] = a["id"].map(id2name); a["win"] = a["id"].map(win)
            a = a[a["name"].isin(tab.index)].dropna(subset=["win"])
            own, nn, p90 = a["name"].map(tab["dc90"]), a["name"].map(n), a["name"].map(tab["p90"])
            mult = model.dc_favourite_multiplier(a["win"], pd.Series(pos, index=a.index))
            sd, thr = PRM.loc[pos, "sd"], PRM.loc[pos, "threshold"]
            hit = (a["defensive_contribution"] >= thr).astype(float)
            for kind, g in VARIANTS:
                w = (nn / g).clip(0, 1) if kind == "capped" else nn / (nn + g)
                w = w.where(nn >= config.DC_SHRINK_MIN_NINETIES, 0.0)
                rate = w * own.fillna(avg) + (1 - w) * avg
                p = model.dc_probability_start(rate * mult, p90, config.DC_SUB_MINUTES[pos], sd, thr)
                rows.append(pd.DataFrame({"variant": f"{kind} {g:g}", "pos": pos, "season": season, "p": p, "hit": hit,
                                          "nineties": nn, "own": own, "avg": avg}))
    shutil.rmtree(TMP, ignore_errors=True)
    R = pd.concat(rows, ignore_index=True)
    R["brier"] = (R.p - R.hit) ** 2
    live = f"capped {config.DC_SHRINK_NINETIES:g}"
    print(f"{int((R.variant == live).sum())} starts scored per variant; live gate = {live}\n")
    g = R.groupby(["pos", "variant"], sort=False).agg(predicted=("p", "mean"), actual=("hit", "mean"), brier=("brier", "mean"))
    g["pred/actual"] = (g.predicted / g.actual).round(2)
    print(g.round(4).to_string())
    # the cases the gate exists for: thin evidence AND an own rate well above the pack
    print("\nTHIN + HIGH: under 10 nineties of evidence and an own rate at least 25% above the position average")
    T = R[(R.nineties < 10) & (R.nineties >= config.DC_SHRINK_MIN_NINETIES) & (R.own >= 1.25 * R.avg)]
    t = T.groupby(["pos", "variant"], sort=False).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"), brier=("brier", "mean"))
    t["pred/actual"] = (t.predicted / t.actual).round(2)
    print(t.round(4).to_string())
    print("\nTHIN + LOW: under 10 nineties and an own rate at least 25% below the average")
    T = R[(R.nineties < 10) & (R.nineties >= config.DC_SHRINK_MIN_NINETIES) & (R.own <= 0.75 * R.avg)]
    t = T.groupby(["pos", "variant"], sort=False).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    t["pred/actual"] = (t.predicted / t.actual).round(2)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
