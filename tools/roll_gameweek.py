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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="the new gameweek to roll to (becomes F1)")
    ap.add_argument("--factor-gw", type=int, default=None,
                    help="archived gameweek to source synthetic player factors from (default: gw-1)")
    a = ap.parse_args()

    step(f"1/2  Shift fixture window to GW{a.gw}",
         PY + [os.path.join(ROOT, "tools", "build_fixtures.py"), "--gw", str(a.gw)])
    synth = PY + [os.path.join(ROOT, "tools", "build_synthetic_gw.py"), "--gw", str(a.gw)]
    if a.factor_gw is not None:
        synth += ["--factor-gw", str(a.factor_gw)]
    step(f"2/2  Seed synthetic F1 markets for GW{a.gw}", synth)

    print(f"\n{'=' * 70}\nRolled to GW{a.gw}. Next:\n"
          f"  - curate inputs/starting_lineups.csv for GW{a.gw} team news (if not done)\n"
          f"  - python -m fpl_pipeline.run                 # GW{a.gw} projections on synthetic markets\n"
          f"  - tools/betway.py                            # when real markets open (overrides synthetic)\n"
          f"  - python -m fpl_pipeline.run --gw {a.gw}            # project + archive once odds are real\n"
          f"{'=' * 70}")


if __name__ == "__main__":
    main()
