# -*- coding: utf-8 -*-
"""Reconcile inputs/starting_lineups.csv to the live FPL roster — exhaustive + name-exact.

    env/Scripts/python tools/sync_lineups_roster.py [--dry-run]

WHY: the master is built by an EXACT-string VLOOKUP of each roster player's name into
starting_lineups.csv (fpl_pipeline/players.py -> io_utils.vlookup, Excel FALSE semantics). A row
whose Player name drifts from the roster spelling (an accent, a transfer, a new signing) silently
resolves to 0.0 with no error. Keeping the file == the roster, keyed on the roster's own name
strings, kills that whole failure class AND gives a holistic view — every PL player is present, most
at 0.0, so nobody is ever out of sight during the weekly sweep.

The roster comes from ingest.load_fpl_players() — the SAME canonical name/team logic the master uses
for its VLOOKUP keys, read straight from the raw FPL data, so it is current even before a pipeline
run. This runs as step 4 of tools/roll_gameweek.py (opt out with --no-roster-sync).

What it does, preserving every existing belief:
  - ADD   any roster player missing from the file as an all-0.0 row (new signings/call-ups surface).
  - DROP  rows whose player is no longer in the roster (departed from the league).
  - REKEY every Player/Team to the exact roster string (fixes accent/spelling/transfer drift).
  - KEEP  each matched player's F1-F8 beliefs untouched.
Start probabilities are NOT re-normalised here (the file stores raw beliefs; normalisation is at
consumption) — but team F1 sums are printed so a belief lost with a dropped player is visible.

The FPL roster is CUMULATIVE — it keeps every player who has been in a PL squad this season, so a
player who has since left the league still appears (with his old club). `inputs/transferred_out.csv`
(columns: name[,team][,note]) is the manual exclusion list: each name is subtracted from the roster
before reconciling, so departed players are neither added nor kept. Add a name here the moment a
player leaves the league.
"""
import argparse
import os
import re
import sys
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import ingest  # noqa: E402

CSV = os.path.join(ROOT, "inputs", "starting_lineups.csv")
TRANSFERRED = os.path.join(ROOT, "inputs", "transferred_out.csv")
COLS = [f"F{i}" for i in range(1, 9)]
_TRANS = str.maketrans({"ø": "o", "Ø": "o", "ß": "ss", "đ": "d", "Đ": "d", "ı": "i",
                        "ł": "l", "Ł": "l", "ð": "d", "þ": "th", "æ": "ae"})


def norm(s):
    s = str(s).translate(_TRANS)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def _transferred_out():
    """Normalised names to subtract from the cumulative roster (players who left the league).
    Returns (excluded_nkeys set, raw_names list) — empty if the file is absent."""
    if not os.path.exists(TRANSFERRED):
        return set(), []
    t = pd.read_csv(TRANSFERRED)
    names = [str(n) for n in t["name"].dropna()]
    return {norm(n) for n in names}, names


def sync(dry_run=False):
    roster = ingest.load_fpl_players()[["name", "team"]].copy()
    roster["nkey"] = roster["name"].map(norm)
    roster["tkey"] = roster["team"].map(norm)

    excluded, excl_names = _transferred_out()
    if excluded:
        present = set(roster["nkey"]) & excluded               # listed names actually in the cumulative roster
        roster = roster[~roster["nkey"].isin(excluded)].copy()  # drop the leavers from the effective roster
        stale = [n for n in excl_names if norm(n) not in present]
        print(f"  transferred_out: {len(excl_names)} listed -> excluded {len(present)} present in roster"
              + (f"; {len(stale)} already absent (prunable): {', '.join(stale)}" if stale else ""))

    d = pd.read_csv(CSV)
    for c in COLS:                                  # tolerate a file missing a column
        if c not in d.columns:
            d[c] = 0.0
    d["nkey"] = d["Player"].map(norm)
    d["tkey"] = d["Team"].map(norm)

    # Belief lookup from the current file: (name,team) first, then name-only if unambiguous.
    by_nt = {(r.nkey, r.tkey): r for r in d.itertuples()}
    by_n = {}
    for r in d.itertuples():
        by_n.setdefault(r.nkey, []).append(r)
    consumed = set()

    def match(rn, rt):
        hit = by_nt.get((rn, rt))                   # exact name+team (survives spelling, same club)
        if hit is not None:
            return hit
        cand = by_n.get(rn, [])                     # name-only (survives a transfer to a new club)
        return cand[0] if len(cand) == 1 else None

    out, added, rekeyed = [], [], []
    for r in roster.itertuples():
        m = match(r.nkey, r.tkey)
        row = {"Player": r.name, "Team": r.team}
        if m is None:
            row.update({c: 0.0 for c in COLS})
            added.append((r.name, r.team))
        else:
            consumed.add(m.Index)
            row.update({c: float(getattr(m, c)) for c in COLS})
            if m.Player != r.name or m.Team != r.team:
                rekeyed.append((m.Player, m.Team, r.name, r.team))
        out.append(row)

    departed = [(r.Player, r.Team, max(float(getattr(r, c)) for c in COLS))
                for r in d.itertuples() if r.Index not in consumed]

    res = pd.DataFrame(out, columns=["Player", "Team"] + COLS)
    res[COLS] = res[COLS].round(3)
    res = res.sort_values(["Team", "F1", "Player"], ascending=[True, False, True]).reset_index(drop=True)

    # ---- report ----
    print(f"  roster {len(roster)} | file {len(d)} -> {len(res)} rows")
    print(f"  +{len(added)} added (now 0.0)   -{len(departed)} departed   ~{len(rekeyed)} re-keyed   "
          f"={len(consumed)} beliefs preserved")
    for old_p, old_t, np_, nt in rekeyed:
        print(f"    re-key: '{old_p}' ({old_t}) -> '{np_}' ({nt})")
    lost = [x for x in departed if x[2] > 0.0]
    for p, t, mx in sorted(lost, key=lambda x: -x[2]):
        print(f"    ⚠ DROPPED with belief {mx:.2f}: {p} ({t}) — team may now sum < 11, re-check")
    if added:
        show = ", ".join(f"{p} ({t})" for p, t in added[:12])
        print(f"    added e.g.: {show}{' …' if len(added) > 12 else ''}")

    if dry_run:
        print("  (--dry-run: no file written)")
        return
    res.drop(columns=[c for c in res.columns if c.startswith("_")], errors="ignore").to_csv(CSV, index=False)
    s = res.groupby("Team")["F1"].sum()
    bad = s[(s - 11).abs() > 0.05]
    print("  team F1 sums: " + ("all 11.00 ✓" if bad.empty else "OFF -> " + bad.round(2).to_dict().__repr__()))
    print(f"  wrote {CSV}")


def main():
    ap = argparse.ArgumentParser(description="Reconcile starting_lineups.csv to the live FPL roster.")
    ap.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    sync(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
