"""Rebuild raw bookmaker goalscorer / assist prices for archived weeks that predate the raw-odds
archive (inputs/raw_odds/, started GW8 2026-27), by inverting the conversion the pipeline applied:

    archived chance = curve( (1 / odds) / MARGIN_PLAYER )

with the curve that was live that week: LONGSHOT_CALIBRATION for goals (unchanged since 18 Aug 2026),
and for assists the regime recorded in config.ASSIST_CURVE_REGIMES (else LONGSHOT_CALIBRATION_ASSIST).

Only GENUINE prices are kept: a real Betway price is a 2-decimal number, so a recovered odds value
that does not land on one (within ODDS_TOL) was never a quoted price - e.g. a player filled in from
his fallback factor - and is dropped. The share that lands is printed per week as a self-check: a
wrong margin or curve would scatter them off the 2-decimal grid.

Output: inputs/raw_odds/<season>/GW<NN>/sportsbet_{goalscorer,assist}_odds_reconstructed.csv plus a
"reconstructed" entry in that folder's _provenance.json. Never overwrites a scraped (non-reconstructed)
file. Weeks whose market was withheld from the archive (synthetic) have nothing to rebuild.

    env/Scripts/python tools/backfill_raw_odds.py [--season 2026-2027] [--to-gw 7] [--dry-run]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, history, markets  # noqa: E402

ODDS_TOL = 0.002
MARKETS = {"goalscorer": ("F1 Score 1+", "sportsbet_goalscorer_odds"),
           "assist": ("F1 Assist", "sportsbet_assist_odds")}
_GRID = np.exp(np.linspace(np.log(1e-5), np.log(0.999), 400_000))     # log grid: fine at long prices


def _knots(market, season, gw):
    if market == "goalscorer":
        return config.LONGSHOT_CALIBRATION
    for reg_season, lo, hi, knots in getattr(config, "ASSIST_CURVE_REGIMES", []):
        if reg_season == season and lo <= gw <= hi:
            return knots
    return config.LONGSHOT_CALIBRATION_ASSIST


def invert(prob, knots):
    """Archived (calibrated, de-margined) chance -> the decimal odds that produced it."""
    curve = markets.longshot_calibrate(pd.Series(_GRID), knots=knots).to_numpy()
    p = pd.to_numeric(prob, errors="coerce").to_numpy(dtype=float)
    raw = np.interp(p, curve, _GRID)                     # curve is monotone increasing
    return pd.Series(1.0 / (raw * config.MARGIN_PLAYER), index=prob.index).where(pd.notna(prob))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=config.SEASON)
    ap.add_argument("--to-gw", type=int, default=7, help="last gameweek to rebuild (the scraped archive starts after it)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    h = pd.read_csv(history.PLAYER_HISTORY_CSV, low_memory=False)
    h = h[(h["Season"] == a.season) & (pd.to_numeric(h["Gameweek"], errors="coerce") <= a.to_gw)]
    print(f"{'GW':>3} {'market':<11}{'archived':>9}{'genuine':>9}{'share':>7}  -> file")
    for gw, wk in h.groupby(pd.to_numeric(h["Gameweek"], errors="coerce")):
        gw = int(gw)
        dest = os.path.join(history.RAW_ODDS_DIR, a.season, f"GW{gw:02d}")
        prov = {}
        for market, (col, stem) in MARKETS.items():
            if col not in wk:
                continue
            w = wk[wk[col].notna() & (pd.to_numeric(wk[col], errors="coerce") > 0)]
            if w.empty:
                print(f"{gw:>3} {market:<11}{0:>9}{'-':>9}{'-':>7}  (withheld / not archived)")
                continue
            odds = invert(w[col], _knots(market, a.season, gw))
            neat = (odds - odds.round(2)).abs() < ODDS_TOL
            out = pd.DataFrame({"player_name": w["Player Name"].values, "team": w["Team"].values,
                                "opponent": w["F1 Opponent"].values, "venue": w["F1 Venue"].values,
                                "odds_decimal": odds.round(2).values})[neat.values]
            fname = f"{stem}_reconstructed.csv"
            print(f"{gw:>3} {market:<11}{len(w):>9}{len(out):>9}{neat.mean():>7.0%}  {os.path.relpath(os.path.join(dest, fname), config.ROOT)}")
            if a.dry_run or out.empty:
                continue
            os.makedirs(dest, exist_ok=True)
            if os.path.exists(os.path.join(dest, f"{stem}.csv")):
                print(f"      scraped {stem}.csv already present - reconstruction written alongside, not over it")
            out.to_csv(os.path.join(dest, fname), index=False)
            prov[fname] = {"state": "reconstructed", "source": "tools/backfill_raw_odds.py",
                           "detail": f"inverted archived {col} (margin {config.MARGIN_PLAYER}, week's curve); "
                                     f"{len(out)} of {len(w)} landed on a 2-decimal price"}
        if prov and not a.dry_run:
            mpath = os.path.join(dest, "_provenance.json")
            doc = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
            doc.update(prov)
            json.dump(doc, open(mpath, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
