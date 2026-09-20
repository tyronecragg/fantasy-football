# -*- coding: utf-8 -*-
"""Roll the pipeline to a new gameweek and seed synthetic F1 markets — one command per week.

    env/Scripts/python tools/roll_gameweek.py --gw N [--factor-gw M]

RUN THIS FIRST, before scraping or editing anything, and NEVER hand-edit inputs/fixtures.csv or
shift the master yourself: build_synthetic_gw's guard checks that the master's F2 really is GW N,
and a manual shift desyncs that and breaks the roll (leaving last week's / fabricated player markets
in place). Precondition: GW N-1 was archived on real odds and nothing was hand-shifted since.

Runs, in order (each a subprocess; a failure stops the roll):
  1. tools/build_fixtures.py --gw N        shift the F1-F8 window to GW N
  2. tools/build_synthetic_gw.py --gw N    carry the master's F2 projection (its forecast for GW N)
                                           onto GW N's F1; every market stamped synthetic in the
                                           provenance manifest. Refuses if master F2 isn't GW N.
  3. shift starting_lineups.csv one GW left F2->F1 ... F8->F7 (F8 held at steady-state), so the
                                           curated beliefs realign to the new window. The new F1 is a
                                           STARTING POINT — re-curate it with GW N team news before
                                           consuming. Skip with --keep-lineups.
  4. tools/sync_lineups_roster.py          reconcile starting_lineups.csv to the live FPL roster:
                                           add new signings/call-ups as 0.0 rows, drop departed
                                           players, re-key every name to the exact roster spelling
                                           (the master's VLOOKUP is exact-match, so drift silently
                                           zeroes a player). Keeps the file exhaustive + name-safe.
                                           Skip with --no-roster-sync.

BEFORE running: update inputs/title_odds.csv / relegation_odds.csv / top6_odds.csv only if outrights
moved materially (they feed the synthetic win probs; most weeks skip). Curate
inputs/starting_lineups.csv for GW N team news.

AFTER running: `python -m fpl_pipeline.run` to see GW N projections on the synthetic markets; run
tools/betway.py + card scrape as the real markets open (each overwrites the markets it prices,
leaving the rest synthetic); `tools/odds_status.py` to check real vs synthetic + fixtures; finally
`python -m fpl_pipeline.run --gw N` at the deadline to project and archive on the real odds.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = [sys.executable, "-X", "utf8"]


def step(title, cmd):
    print(f"\n{'=' * 70}\n{title}\n  {' '.join(cmd)}\n{'=' * 70}")
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit(f"roll aborted: step failed ({title})")


def shift_lineups():
    """Slide starting_lineups.csv one GW left: F2->F1 ... F8->F7, F8 held at last week's steady-state.
    Aligns the curated beliefs to the new window (old F2 was already the forecast for the new F1).
    The new F1 is a STARTING POINT — re-curate it with the new GW's team news before consuming."""
    import csv
    path = os.path.join(ROOT, "inputs", "starting_lineups.csv")
    cols = [f"F{i}" for i in range(1, 9)]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        header = list(rows[0].keys()) if rows else ["Player", "Team", *cols]
    for r in rows:
        vals = [r[c] for c in cols]
        for i in range(7):
            r[cols[i]] = vals[i + 1]   # F(i+1) <- old F(i+2)
        r[cols[7]] = vals[7]           # F8 carries old F8 (steady-state)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  shifted {len(rows)} rows: F2->F1 ... F8->F7, F8 held. Re-curate the new F1 with team news.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="the new gameweek to roll to (becomes F1)")
    ap.add_argument("--factor-gw", type=int, default=None,
                    help="archived gameweek to source synthetic player factors from (default: gw-1)")
    ap.add_argument("--keep-lineups", action="store_true",
                    help="do NOT shift starting_lineups.csv (default: slide F2->F1 ... F8->F7)")
    ap.add_argument("--no-roster-sync", action="store_true",
                    help="do NOT reconcile starting_lineups.csv to the live FPL roster")
    a = ap.parse_args()

    step(f"1/4  Shift fixture window to GW{a.gw}",
         PY + [os.path.join(ROOT, "tools", "build_fixtures.py"), "--gw", str(a.gw)])
    synth = PY + [os.path.join(ROOT, "tools", "build_synthetic_gw.py"), "--gw", str(a.gw)]
    if a.factor_gw is not None:
        synth += ["--factor-gw", str(a.factor_gw)]
    step(f"2/4  Seed synthetic F1 markets for GW{a.gw}", synth)
    if a.keep_lineups:
        print("\n(skipping starting_lineups.csv shift; --keep-lineups)")
    else:
        print(f"\n{'=' * 70}\n3/4  Shift starting_lineups.csv one GW left\n{'=' * 70}")
        shift_lineups()
    if a.no_roster_sync:
        print("\n(skipping roster sync; --no-roster-sync)")
    else:
        step(f"4/4  Reconcile starting_lineups.csv to the live FPL roster",
             PY + [os.path.join(ROOT, "tools", "sync_lineups_roster.py")])

    print(f"\n{'=' * 70}\nRolled to GW{a.gw}. Next:\n"
          f"  - RE-CURATE inputs/starting_lineups.csv F1 for GW{a.gw} team news (the shift gave you last\n"
          f"    week's forecast as a starting point — sweep it against actual GW{a.gw} news)\n"
          f"  - python -m fpl_pipeline.run                 # GW{a.gw} projections on synthetic markets\n"
          f"  - tools/betway.py                            # when real markets open (overrides synthetic)\n"
          f"  - python -m fpl_pipeline.run --gw {a.gw}            # project + archive once odds are real\n"
          f"  - compile the teams (source-accuracy archive): capture RotoWire + FFS + FFP at the\n"
          f"    deadline, seed ours, then score after GW{a.gw} plays — docs/SOURCE_ACCURACY_CAPTURE.md\n"
          f"{'=' * 70}")


if __name__ == "__main__":
    main()
