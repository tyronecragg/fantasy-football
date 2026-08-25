"""One-command weekly update for the FPL pipeline.

Phase 1 (default) - gather the week's data, then pause for lineup curation:
    python weekly_update.py
  1. pull the latest FPL data (git)
  2. scrape Betway odds (tools/betway.py - plain requests, no VPN needed)
  3. stage the FFS predicted lineups + team news for curation

  -> then review inputs/ffs_team_news.md and the printed diff with Claude, update
     inputs/starting_lineups.csv, and run phase 2.

Phase 2 (--resume) - rebuild everything downstream of the curated lineups:
    python weekly_update.py --resume --gw N
  4. rebuild the F1-F8 fixture window for GW N
  5. freeze this week's predicted XIs (FFS + our curation) into the source-history ledger
  6. rebuild projections (records GW N archives; guarded against synthetic odds)
  7. run the transfer optimiser (PuLP venv)
  8. log chip values + forward F1..F8 chip radar, then score LAST week's frozen predictions against the actual XIs

Flags: --gw N (required to archive; omit to rebuild without touching archives),
       --skip-odds (skip the Betway scrape; --skip-sportsbet still accepted),
       --force-archive (override the synthetic-odds guard).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-X", "utf8"]
VENV_PY = os.path.join(ROOT, "env", "Scripts", "python.exe")

# Files Excel tends to hold open; a locked one aborts the phase before any half-writes
LOCK_CHECKS = {
    1: ["sportsbet/sportsbet_win_draw_win_odds.csv", "inputs/ffs_predicted_lineups.csv"],
    2: ["outputs/13_players_master.csv", "inputs/starting_lineups.csv", "inputs/fixtures.csv"],
}

results = []


def report(name, ok, detail=""):
    results.append((name, ok, detail))
    tag = "[OK]  " if ok else "[WARN]"
    print(f"\n{tag} {name}" + (f" - {detail}" if detail else ""))


def step(name, cmd, fatal=True):
    print(f"\n{'=' * 70}\nSTEP: {name}\n  {' '.join(cmd)}\n{'=' * 70}")
    proc = subprocess.run(cmd, cwd=ROOT)
    ok = proc.returncode == 0
    report(name, ok, "" if ok else f"exit code {proc.returncode}")
    if not ok and fatal:
        summary_and_exit(1)
    return ok


def check_locks(phase):
    locked = []
    for rel in LOCK_CHECKS[phase]:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "a", encoding="utf-8"):
                pass
        except PermissionError:
            locked.append(rel)
    if locked:
        print("These files are locked (open in Excel?) - close them and rerun:")
        for rel in locked:
            print(f"  {rel}")
        summary_and_exit(1)


def summary_and_exit(code):
    if results:
        print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
        for name, ok, detail in results:
            print(f"  {'[OK]  ' if ok else '[WARN]'} {name}" + (f" - {detail}" if detail else ""))
    sys.exit(code)


def phase1(argv):
    check_locks(1)

    step("Pull FPL data", ["git", "-C", os.path.join(ROOT, "fpl_data", "FPL-Core-Insights"), "pull"],
         fatal=False)

    if "--skip-odds" in argv or "--skip-sportsbet" in argv:   # --skip-sportsbet kept as an alias
        report("Betway scrape", True, "skipped")
    else:
        odds_csv = os.path.join(ROOT, "sportsbet", "sportsbet_win_draw_win_odds.csv")
        before = os.path.getmtime(odds_csv) if os.path.exists(odds_csv) else 0
        step("Betway scrape", PY + [os.path.join(ROOT, "tools", "betway.py")], fatal=False)
        after = os.path.getmtime(odds_csv) if os.path.exists(odds_csv) else 0
        if after == before:
            report("Betway freshness", False, "odds files were NOT refreshed - check the scrape output")

    step("Stage FFS lineups + team news", PY + [os.path.join(ROOT, "starting_lineups.py")], fatal=False)
    step("Injury cross-check vs FPL flags", PY + [os.path.join(ROOT, "tools", "injury_check.py")],
         fatal=False)

    print(f"\n{'=' * 70}")
    print("PAUSED FOR CURATION")
    print("  1. Review the diff above and inputs/ffs_team_news.md with Claude")
    print("  2. Update inputs/starting_lineups.csv (graded start probabilities)")
    print("  3. Continue with:  python weekly_update.py --resume --gw N")
    print(f"{'=' * 70}")
    summary_and_exit(0)


def phase2(argv):
    check_locks(2)
    gw = argv[argv.index("--gw") + 1] if "--gw" in argv else None

    if gw:
        step("Build fixture window", PY + [os.path.join(ROOT, "tools", "build_fixtures.py"), "--gw", gw])
    else:
        report("Build fixture window", True, "skipped (no --gw; keeping the current F1-F8 window)")

    # Freeze this week's predictions BEFORE the pipeline can move on - FFS staging + our curated XI.
    # External sources (RotoWire, All About FPL, the deadline sweep) are captured during curation
    # via tools/source_history.capture(); see inputs/curation_sources.md "Earning trust".
    if gw:
        step("Freeze source predictions (FFS + our curation)",
             PY + [os.path.join(ROOT, "tools", "source_history.py"), "--seed-ffs", "--seed-ours", "--gw", gw],
             fatal=False)

    pipeline = PY + ["-m", "fpl_pipeline.run"]
    if gw:
        pipeline += ["--gw", gw]
        if "--force-archive" in argv:
            pipeline += ["--force-archive"]
    else:
        print("\nNo --gw given: projections rebuild but archives are not recorded.")
    step("Rebuild projections", pipeline)

    optimiser_py = [VENV_PY] if os.path.exists(VENV_PY) else [sys.executable]
    step("Transfer optimiser", optimiser_py + ["-X", "utf8", os.path.join(ROOT, "optimisation.py")],
         fatal=False)

    chip_cmd = optimiser_py + ["-X", "utf8", os.path.join(ROOT, "tools", "chip_history.py"), "--radar"]
    if gw:
        chip_cmd += ["--gw", gw]
    step("Log chip values + forward F1..F8 chip radar (Bench Boost / Triple Captain / Wildcard / Free Hit)",
         chip_cmd, fatal=False)
    # This auto-run uses the DEFAULT 1 free transfer and £0 bank - the wildcard/free-hit deltas (and the
    # radar's Free Hit column) are only right when those match reality. Re-run by hand with your real
    # numbers to overwrite the row and refresh the radar.
    report("Chip values - MANUAL RE-RUN REMINDER", True,
           f"deltas + radar above assume 1 free transfer / £0 bank; if yours differ, re-run: "
           f"env\\Scripts\\python tools/chip_history.py --gw {gw or 'N'} --free-transfers <FT> --bank <BANK> --radar")

    # Now that GW N is building, GW N-1 has played - score last week's frozen predictions against
    # the actual XIs. Skips cleanly (fatal=False) if the actuals aren't in the FPL data yet.
    if gw and int(gw) > 1:
        step("Score last week's predictions vs actual XIs",
             PY + [os.path.join(ROOT, "tools", "source_history.py"), "--score", "--gw", str(int(gw) - 1)],
             fatal=False)
    summary_and_exit(0)


if __name__ == "__main__":
    if "--resume" in sys.argv:
        phase2(sys.argv)
    else:
        phase1(sys.argv)
