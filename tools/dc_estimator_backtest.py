# -*- coding: utf-8 -*-
"""Mean, median, trimmed mean, or the player's own hit rate? Walk-forward test of how to summarise a
player's DefCon history.

    env/Scripts/python tools/dc_estimator_backtest.py

For every target gameweek (2025-26 GW16-29, 2026-27 GW2-5) each player's history is rebuilt from the
per-gameweek dumps BEFORE it: appearances of 45+ minutes as DC per 90, this season weighted 2x, up to the
last 19 appearances of the prior season at 1x, each weighted by minutes. All estimators then share ONE
simplified pipeline (evidence gate 4 nineties toward the population average, full-90/subbed mixture,
defender favourite curve; no mover discount / neutralisation / external priors) so ONLY the summary differs:
    mean        minutes-weighted mean per 90 (what the live model uses)
    median      weighted median of the per-90 match values
    trimmed     weighted mean after clipping each match to the player's own 20th-80th percentile
    hit-rate k  his own share of full-90 games at/over the threshold, shrunk with k pseudo-games at the
                position's hit rate (no bell curve at all for the full-90 leg)
    blend       average of `mean` and `hit-rate 8`
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from fpl_pipeline import config, model  # noqa: E402

DATA = os.path.dirname(config.FPL_DATA_DIR.rstrip("\\/"))
PRM = pd.read_csv(os.path.join(config.INPUTS_DIR, "dc_params.csv")).set_index("position")
SEASONS = ["2025-2026", "2026-2027"]


def load():
    fr = []
    for s in SEASONS:
        pl = pd.read_csv(os.path.join(DATA, s, "players.csv"), usecols=["player_id", "player_code", "position"])
        g = pd.concat([pd.read_csv(p, usecols=["id", "gw", "minutes", "starts", "defensive_contribution", "tackles",
                                               "clearances_blocks_interceptions"])
                       for p in glob.glob(os.path.join(DATA, s, "By Gameweek", "GW*", "player_gameweek_stats.csv"))])
        g = g.drop_duplicates(["id", "gw"]).merge(pl, left_on="id", right_on="player_id")
        g["season"] = s; g["pos"] = g["position"].map(config.POSITION_MAP)
        # UPSTREAM BUG (found 2026-09-21): in 2025-26 GW2-GW10 the defender `defensive_contribution` column
        # wrongly includes recoveries (GW2 is worse). Defenders' DefCon is tackles + clearances/blocks/
        # interceptions, which the component columns give correctly in every gameweek - use that, and drop GW2.
        is_def = g["pos"] == "DEF"
        g.loc[is_def, "defensive_contribution"] = (g["tackles"] + g["clearances_blocks_interceptions"])[is_def]
        if s == "2025-2026":
            g = g[g["gw"] != 2]
        fr.append(g)
    d = pd.concat(fr, ignore_index=True)
    d = d[d.pos.isin(["DEF", "MID"])]
    d["t"] = d.season.map({s: i for i, s in enumerate(SEASONS)}) * 100 + d.gw
    return d


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def summarise(h, thr):
    """one row per player_code from his weighted history"""
    out = {}
    for code, g in h.groupby("player_code"):
        x, w = g.x.values, g.w.values
        lo, hi = wquantile(x, w, 0.2), wquantile(x, w, 0.8)
        f = g[g.minutes >= 85]
        out[code] = {"mean": float(np.average(x, weights=w)), "median": wquantile(x, w, 0.5),
                     "trimmed": float(np.average(np.clip(x, lo, hi), weights=w)),
                     "n_eff": float(g.nineties.sum()), "hit_w": float((f.rw * (f.defensive_contribution >= thr)).sum()),
                     "full_w": float(f.rw.sum()), "starts_w": float(g.loc[g.starts == 1, "rw"].sum()),
                     "full_starts_w": float(g.loc[(g.starts == 1) & (g.minutes >= 90), "rw"].sum())}
    return pd.DataFrame.from_dict(out, orient="index")


def main():
    d = load()
    arch = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False,
                       usecols=["Season", "Gameweek", "player_id", "F1 Win"])
    for c in ["Gameweek", "player_id", "F1 Win"]:
        arch[c] = pd.to_numeric(arch[c], errors="coerce")
    rows = []
    targets = [("2025-2026", g) for g in range(16, 30)] + [("2026-2027", g) for g in range(2, 6)]
    for season, gw in targets:
        win = (arch[(arch.Season == season) & (arch.Gameweek == gw)].dropna(subset=["player_id"])
               .drop_duplicates("player_id").set_index("player_id")["F1 Win"])
        act = d[(d.season == season) & (d.gw == gw) & (d.starts == 1)].copy()
        act["win"] = act["id"].map(win); act = act.dropna(subset=["win"])
        if act.empty:
            continue
        cur = d[(d.season == season) & (d.gw < gw) & (d.minutes >= 45)].copy(); cur["rec"] = 2.0
        pri = d[(d.season < season) & (d.minutes >= 45)].sort_values("t").groupby("player_code").tail(19).copy(); pri["rec"] = 1.0
        h = pd.concat([cur, pri]); h["x"] = h.defensive_contribution * 90.0 / h.minutes
        h["nineties"] = h.minutes / 90.0; h["w"] = h.rec * h.nineties; h["rw"] = h.rec
        for pos in ["DEF", "MID"]:
            thr, sd, sub = PRM.loc[pos, "threshold"], PRM.loc[pos, "sd"], config.DC_SUB_MINUTES[pos]
            S = summarise(h[h.pos == pos], thr)
            a = act[act.pos == pos].copy()
            s = S.reindex(a.player_code.values); s.index = a.index
            n = s.n_eff.fillna(0.0)
            gate = (n / 4.0).clip(0, 1).where(n >= config.DC_SHRINK_MIN_NINETIES, 0.0)
            k = float(config.DC_MINUTES_SHRINK_STARTS)
            p90 = (s.full_starts_w.fillna(0) + k * config.DC_FULL90_POSITION_AVG[pos]) / (s.starts_w.fillna(0) + k)
            mult = model.dc_favourite_multiplier(a["win"], pd.Series(pos, index=a.index))
            hit = (a.defensive_contribution >= thr).astype(float)
            P = lambda mu: pd.Series(1 - norm.cdf(thr, np.asarray(mu, dtype=float), sd), index=a.index)
            mix = lambda rate: p90 * P(rate * mult) + (1 - p90) * P(rate * mult * sub / 90.0)
            rates = {}
            for est in ["mean", "median", "trimmed"]:
                pop = float(np.average(S.loc[S.n_eff >= 4, est], weights=S.loc[S.n_eff >= 4, "n_eff"]))
                rates[est] = gate * s[est].fillna(pop) + (1 - gate) * pop
                rows.append(pd.DataFrame({"est": est, "pos": pos, "season": season, "p": mix(rates[est]), "hit": hit, "n": n}))
            pos_hit = S.hit_w.sum() / max(S.full_w.sum(), 1e-9)
            base_full = P(rates["mean"]).clip(lower=1e-4)
            for kk in [4.0, 8.0, 16.0]:
                pf = (s.hit_w.fillna(0) + kk * pos_hit) / (s.full_w.fillna(0) + kk)     # his own full-90 hit rate, shrunk
                # fixture + subbed-off legs borrow only the RATIO from the bell curve
                p = p90 * pf * (P(rates["mean"] * mult) / base_full) + (1 - p90) * pf * (P(rates["mean"] * mult * sub / 90.0) / base_full)
                p = p.clip(0, 1)
                rows.append(pd.DataFrame({"est": f"hit-rate {kk:g}", "pos": pos, "season": season, "p": p, "hit": hit, "n": n}))
                if kk == 8.0:
                    rows.append(pd.DataFrame({"est": "blend", "pos": pos, "season": season, "p": 0.5 * p + 0.5 * mix(rates["mean"]), "hit": hit, "n": n}))
    R = pd.concat(rows, ignore_index=True); R["brier"] = (R.p - R.hit) ** 2
    print(f"{int((R.est == 'mean').sum())} starts scored per estimator\n")
    g = R.groupby(["pos", "est"], sort=False).agg(predicted=("p", "mean"), actual=("hit", "mean"), brier=("brier", "mean"))
    g["pred/actual"] = (g.predicted / g.actual).round(2)
    print(g.round(4).to_string())
    print("\nplayers with under 8 nineties of history (where one big game matters most):")
    t = R[(R.n < 8) & (R.n >= 1)].groupby(["pos", "est"], sort=False).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"), brier=("brier", "mean"))
    print(t.round(4).to_string())
    print("\nby season (Brier):")
    print(R.pivot_table(index=["pos", "est"], columns="season", values="brier", aggfunc="mean", sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
