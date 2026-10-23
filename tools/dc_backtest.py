"""Walk-forward backtest of the DefCon model variants (kept 2026-09-21; run: env/Scripts/python tools/dc_backtest.py). For every target gameweek, rebuild the DefCon inputs
from ONLY the data available before it (truncated copies of the FPL dumps in a temp dir, the real ingest
code run against them), price each actual starter, and score against whether he cleared the threshold.
Variants: A flat per-90 | B + minutes per start | C + favourite curve | D + fixture-mix neutralisation."""
import glob
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Tyrone\Documents\fantasy_premier_league"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest, model, players, rating  # noqa: E402

DATA = os.path.dirname(config.FPL_DATA_DIR.rstrip("\\/"))
PRM = pd.read_csv(os.path.join(config.INPUTS_DIR, "dc_params.csv")).set_index("position")
CURVE = dict(config.DC_FAVOURITE_CURVE)
ARCH = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False,
                   usecols=["Season", "Gameweek", "player_id", "F1 Win"])
for c in ["Gameweek", "player_id", "F1 Win"]:
    ARCH[c] = pd.to_numeric(ARCH[c], errors="coerce")
MATCHES = rating.load_matches([rating.season_code(s) for s in ["2025-2026", "2026-2027"]])
TMP = tempfile.mkdtemp(prefix="dc_asof_")


def copy_season(season, upto=None):
    """temp copy of the files ingest reads; upto=N keeps only data from gameweeks < N"""
    src, dst = os.path.join(DATA, season), os.path.join(TMP, season)
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


def cutoff_date(season, gw):
    m = pd.read_csv(os.path.join(DATA, season, "By Gameweek", f"GW{gw}", "matches.csv"))
    m = m[m["match_id"].astype(str).str.contains("-prem-")]
    col = [c for c in m.columns if "kickoff" in c.lower()][0]
    return pd.to_datetime(m[col], utc=True).min().tz_localize(None).normalize()


def names_by_id(season):
    pl = pd.read_csv(os.path.join(DATA, season, "players.csv"))
    nm = ingest._apply_name_changes(pl["first_name"] + " " + pl["second_name"])
    return dict(zip(pl["player_id"], nm)), dict(zip(pl["player_id"], pl["position"].map(config.POSITION_MAP)))


P90_AVG = {"DEF": 0.77, "MID": 0.48}


def full90_share(season, dst, prior_season_dir, id2name_cur):
    """name -> shrunk P(plays the full 90 | starts), from starts before the target (2x recency) + <=10 prior starts, k=7"""
    def agg(d, id2name):
        fr = [pd.read_csv(p, usecols=["id", "gw", "minutes", "starts"]) for p in glob.glob(os.path.join(d, "By Gameweek", "GW*", "player_gameweek_stats.csv"))]
        if not fr:
            return pd.DataFrame(columns=["n", "f"])
        g = pd.concat(fr).drop_duplicates(["id", "gw"]); g = g[g.starts == 1]
        g["name"] = g["id"].map(id2name); g["full"] = (g.minutes >= 90).astype(float)
        return g.groupby("name").agg(n=("full", "size"), f=("full", "sum"))
    cur = agg(dst, id2name_cur)
    pri = agg(prior_season_dir, names_by_id(os.path.basename(prior_season_dir))[0]) if prior_season_dir else pd.DataFrame(columns=["n", "f"])
    return cur, pri


SUB_MIN = {"DEF": 66.0, "MID": 68.0}     # mean minutes of starts that end before 90 (measured below)
VARIANTS = {"A flat per-90": (False, False, False), "B + minutes": (True, False, False),
            "C + favourite curve": (True, True, False), "D + neutralised (live)": (True, True, True)}
_all = pd.concat([pd.read_csv(p, usecols=["id", "minutes", "starts"]) for p in glob.glob(os.path.join(DATA, "2025-2026", "By Gameweek", "GW*", "player_gameweek_stats.csv"))])
_pl = pd.read_csv(os.path.join(DATA, "2025-2026", "players.csv")).set_index("player_id")["position"].map(config.POSITION_MAP)
_all = _all[(_all.starts == 1) & (_all.minutes < 90)]; _all["pos"] = _all["id"].map(_pl)
for _p in ["DEF", "MID"]:
    SUB_MIN[_p] = float(_all[_all.pos == _p].minutes.mean())
print("mean minutes of starts ending before 90:", {k: round(v, 1) for k, v in SUB_MIN.items()})
rows = []
targets = [("2025-2026", g) for g in range(16, 30)] + [("2026-2027", g) for g in range(2, 6)]
copy_season("2025-2026")                                   # full prior season, used by the 2026-27 targets
for season, gw in targets:
    act_p = os.path.join(DATA, season, "By Gameweek", f"GW{gw}", "player_gameweek_stats.csv")
    if not os.path.exists(act_p):
        continue
    act = pd.read_csv(act_p, usecols=["id", "minutes", "starts", "defensive_contribution"]).drop_duplicates("id")
    act = act[act["starts"] == 1]
    win = ARCH[(ARCH.Season == season) & (ARCH.Gameweek == gw)].dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["F1 Win"]
    if win.dropna().empty:
        continue
    if season == "2025-2026":
        root = os.path.join(TMP, "solo"); os.makedirs(root, exist_ok=True)      # no prior sibling for 2025-26
        dst = os.path.join(root, season)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        copy_season(season, upto=gw); shutil.move(os.path.join(TMP, season), dst)
        copy_season("2025-2026")                                                # restore the full copy
    else:
        copy_season(season, upto=gw); dst = os.path.join(TMP, season)
    config.SEASON, config.FPL_DATA_DIR = season, dst
    cut = cutoff_date(season, gw)
    ingest._load_favourite_mix = (lambda seasons, _c=cut: ingest.favourite_mix(MATCHES[MATCHES["date"] < _c])
                                  if (config.DC_NEUTRALISE_FIXTURE_MIX and config.DC_FAVOURITE_CURVE) else {})
    id2name, id2pos = names_by_id(season)
    for label, (mins, curve, neutral) in VARIANTS.items():
        config.DC_MINUTES_PER_START, config.DC_FAVOURITE_CURVE, config.DC_NEUTRALISE_FIXTURE_MIX = mins, (CURVE if curve else {}), neutral
        dc = ingest.load_defensive_contributions()
        for pos in ["DEF", "MID"]:
            tab = players._dc_table(dc[pos], PRM.loc[pos], improved=True).drop_duplicates("name").set_index("name")
            a = act[act["id"].map(id2pos) == pos].copy()
            a["rate"] = a["id"].map(id2name).map(tab["rate_eff"])
            a["win"] = a["id"].map(win)
            a = a.dropna(subset=["rate", "win"])
            mult = model.dc_favourite_multiplier(a["win"], pd.Series(pos, index=a.index))
            a["p"] = model.dc_probability(a["rate"] * mult, PRM.loc[pos, "sd"], PRM.loc[pos, "threshold"])
            a["hit"] = (a["defensive_contribution"] >= PRM.loc[pos, "threshold"]).astype(float)
            rows.append(a.assign(variant=label, pos=pos, season=season, gw=gw)[["variant", "pos", "season", "gw", "win", "p", "hit"]])
    # mixture variants: P = p90 * P(hit | full rate) + (1 - p90) * P(hit | rate * sub_minutes / 90)
    cur, pri = full90_share(season, dst, os.path.join(TMP, "2025-2026") if season == "2026-2027" else None, id2name)
    for neutral in [False, True]:
        for lo, end_mult in [(0.55, None), (0.55, 0.70), (0.50, 0.70), (0.50, 0.65), (0.45, 0.70), (0.45, 0.65), (0.45, 0.60), (0.40, 0.65)]:
            if neutral and end_mult is None:
                continue
            crv = {} if end_mult is None else {"DEF": (lo, 0.85, end_mult)}
            config.DC_MINUTES_PER_START, config.DC_FAVOURITE_CURVE, config.DC_NEUTRALISE_FIXTURE_MIX = False, crv, neutral
            dc = ingest.load_defensive_contributions()
            for pos in (["DEF", "MID"] if end_mult is None else ["DEF"]):
                tab = players._dc_table(dc[pos], PRM.loc[pos], improved=True).drop_duplicates("name").set_index("name")
                a = act[act["id"].map(id2pos) == pos].copy()
                a["name"] = a["id"].map(id2name); a["rate"] = a["name"].map(tab["rate_eff"]); a["win"] = a["id"].map(win)
                a = a.dropna(subset=["rate", "win"])
                nc, fc = a["name"].map(cur["n"]).fillna(0), a["name"].map(cur["f"]).fillna(0)
                npri, fpri = a["name"].map(pri["n"]).fillna(0), a["name"].map(pri["f"]).fillna(0)
                cap = npri.clip(upper=10); fpri = (fpri * cap / npri).where(npri > 0, 0.0)
                p90 = (2 * fc + fpri + 7 * P90_AVG[pos]) / (2 * nc + cap + 7)
                a["hit"] = (a["defensive_contribution"] >= PRM.loc[pos, "threshold"]).astype(float)
                mult = model.dc_favourite_multiplier(a["win"], pd.Series(pos, index=a.index))
                sd, thr = PRM.loc[pos, "sd"], PRM.loc[pos, "threshold"]
                pf = model.dc_probability(a["rate"] * mult, sd, thr); ps_ = model.dc_probability(a["rate"] * mult * SUB_MIN[pos] / 90.0, sd, thr)
                lab = "E mixture" + ("" if end_mult is None else f" + curve {lo:.2f}->{end_mult:.2f}" + (" + neutral" if neutral else ""))
                rows.append(a.assign(p=p90 * pf + (1 - p90) * ps_, variant=lab, pos=pos, season=season, gw=gw)[["variant", "pos", "season", "gw", "win", "p", "hit"]])
shutil.rmtree(TMP, ignore_errors=True)
R = pd.concat(rows, ignore_index=True)
R["brier"] = (R.p - R.hit) ** 2
print(f"targets: {R[['season','gw']].drop_duplicates().shape[0]} gameweeks (2025-26 GW16-29, 2026-27 GW2-5); starts scored per variant: {len(R[R.variant=="A flat per-90"])}")
g = R.groupby(["pos", "variant"]).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"), brier=("brier", "mean"))
g["pred/actual"] = (g.predicted / g.actual).round(2)
print(g.round(4).to_string())
D = R[R.pos == "DEF"].copy(); D["band"] = pd.cut(D.win, [0, 0.55, 0.7, 1.0], labels=["<=0.55", "0.55-0.7", ">0.7"])
rows_ = []
for (v, b), x in D.groupby(["variant", "band"], observed=True):
    rows_.append({"variant": v, "band": b, "ratio": round(x.p.mean() / x.hit.mean(), 2)})
print("DEF predicted/actual by win band:")
print(pd.DataFrame(rows_).pivot(index="variant", columns="band", values="ratio").to_string())
