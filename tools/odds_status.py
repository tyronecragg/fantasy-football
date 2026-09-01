"""Print the odds provenance manifest: what is real vs synthetic right now, per market.

    python tools/odds_status.py

Reads sportsbet/_provenance.json (fpl_pipeline/provenance.py). Markets are stamped by the tools
that write them: build_synthetic_gw (synthetic), betway.py (real), bet365/ladbrokes cards. A market
with no entry shows 'unknown' - run tools/betway.py to populate it.

It ALSO cross-checks each team-bearing market file's FIXTURES against inputs/fixtures.csv and flags
any that point at the wrong gameweek - the failure the provenance state alone can't catch (a file
can be genuinely 'real' yet left over from last week, e.g. saves/bookings when Betway hadn't priced
this week's markets). Player-prop files keyed only by match_id carry no team names, so are skipped."""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, provenance  # noqa: E402
from fpl_pipeline.names import apply_team_names  # noqa: E402

MARK = {"real": "[REAL]", "synthetic": "[SYNTH]", "derived": "[deriv]", "unknown": "[ ?? ]"}


def _canon(names):
    """Canonicalise a list of team names to the fixtures.csv spelling (AFC Bournemouth -> Bournemouth)."""
    return list(apply_team_names(pd.Series([str(n).strip() for n in names])))


def _pairs_from_df(df):
    """Best-effort set of {teamA, teamB} fixture pairs from a market file, whatever its schema.
    Returns None when the file carries no team/fixture columns (player-prop files keyed by match_id)."""
    cols = {c.lower(): c for c in df.columns}
    if "home_team" in cols and "away_team" in cols:
        a, b = df[cols["home_team"]], df[cols["away_team"]]
    elif "team" in cols and "opponent" in cols:
        a, b = df[cols["team"]], df[cols["opponent"]]
    else:                                            # try a "A vs. B" / "A v B" match column
        mcol = next((cols[c] for c in ("match", "match_name") if c in cols), None)
        if mcol is None:
            return None
        split = df[mcol].astype(str).str.split(r"\s+vs?\.?\s+", regex=True, expand=True)
        if split.shape[1] < 2:
            return None
        a, b = split[0], split[1]
    ca, cb = _canon(a), _canon(b)
    return {frozenset((x, y)) for x, y in zip(ca, cb) if x and y and x != "nan" and y != "nan"}


def _gw_windows():
    """From inputs/fixtures.csv: {gw_number: set of {teamA,teamB} pairs} for each F1..F8 opponent column."""
    fx = pd.read_csv(os.path.join(config.ROOT, "inputs", "fixtures.csv"))
    home = _canon(fx.iloc[:, 0])
    out = {}
    for col in fx.columns[1:]:
        m = re.match(r"GW(\d+)\s*Opponent", str(col), re.I)   # skip the interleaved 'GW3 Venue' columns
        if not m:
            continue
        away = _canon(fx[col])
        out[int(m.group(1))] = {frozenset((h, a)) for h, a in zip(home, away)
                                if h and a and a != "nan"}
    return out


def _fixture_check(markets):
    windows = _gw_windows()
    cur_gw = min(windows) if windows else None      # F1 column = current gameweek
    flags = []
    for fname in provenance.FRIENDLY:
        path = os.path.join(config.SPORTSBET_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            pairs = _pairs_from_df(pd.read_csv(path))
        except Exception:
            pairs = None
        if not pairs:                               # unparseable / player-prop / placeholder
            continue
        best_gw, best_overlap = None, 0.0
        for gw, wp in windows.items():
            ov = len(pairs & wp) / len(pairs)
            if ov > best_overlap:
                best_gw, best_overlap = gw, ov
        expected = cur_gw + 1 if fname.endswith("_f2.csv") else cur_gw   # _f2 files hold the NEXT gameweek (F2)
        if best_gw != expected or best_overlap < 0.5:
            where = f"looks like GW{best_gw}" if best_gw and best_overlap >= 0.5 else "matches no fixture in the current window"
            flags.append((f"{provenance.FRIENDLY[fname]} (expected GW{expected})", where))
    return cur_gw, flags


def main():
    doc = provenance.status()
    markets = doc.get("markets", {})
    print(f"\nOdds provenance - GW{doc.get('gw')}  (updated {doc.get('updated')})\n")
    # show every known sportsbet market in a stable order, even if unstamped
    for fname, label in provenance.FRIENDLY.items():
        e = markets.get(fname)
        st = e["state"] if e else "unknown"
        src = f"{e['source']}" + (f" - {e['detail']}" if e.get("detail") else "") if e else "no entry (run betway)"
        print(f"  {MARK.get(st, st):<8} {label:<18} {src}")
    real = sum(1 for e in markets.values() if e.get("state") == "real")
    synth = [provenance.FRIENDLY.get(f, f) for f, e in markets.items() if e.get("state") == "synthetic"]
    print(f"\n  {real} market(s) real; still synthetic: {', '.join(synth) or 'none'}")

    cur_gw, flags = _fixture_check(markets)
    if flags:
        print("\n  !! FIXTURE MISMATCH - these files don't match their expected gameweek (stale? re-scrape/roll):")
        for label, where in flags:
            print(f"       {label:<28} {where}")
    else:
        print(f"\n  fixture check: every market matches its gameweek (F1 = GW{cur_gw}, F2 = GW{cur_gw + 1}).")
    print("  --gw archiving should record only real markets (provenance.is_real).\n")


if __name__ == "__main__":
    main()
