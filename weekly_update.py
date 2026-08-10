"""One-command weekly update for the FPL pipeline.

Phase 1 (default) - gather the week's data, then pause for lineup curation:
    python weekly_update.py
  1. pull the latest FPL data (git)
  2. scrape Sportsbet odds (skipped with a warning if the API looks geo-blocked - VPN)
  3. stage the FFS predicted lineups + team news for curation

  -> then review inputs/ffs_team_news.md and the printed diff with Claude, update
     inputs/starting_lineups.csv, and run phase 2.

Phase 2 (--resume) - rebuild everything downstream of the curated lineups:
    python weekly_update.py --resume --gw N
  4. rebuild the F1-F8 fixture window for GW N
  5. rebuild projections (records GW N archives; guarded against synthetic odds)
  6. run the transfer optimiser (PuLP venv)

Flags: --gw N (required to archive; omit to rebuild without touching archives),
       --skip-sportsbet, --force-scrape (ignore the VPN preflight),
       --force-archive (override the synthetic-odds guard).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-X", "utf8"]
VENV_PY = os.path.join(ROOT, "env", "Scripts", "python.exe")
SPORTSBET_API = "https://www.sportsbet.com.au/apigw/sportsbook-sports"

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


def sportsbet_reachable():
    try:
        import requests
        status = requests.get(SPORTSBET_API, timeout=8).status_code
        return status != 403, f"HTTP {status}"
    except Exception as exc:
        return False, str(exc)


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

    if "--skip-sportsbet" in argv:
        report("Sportsbet scrape", True, "skipped (--skip-sportsbet)")
    else:
        reachable, detail = sportsbet_reachable()
        if not reachable and "--force-scrape" not in argv:
            report("Sportsbet scrape", False,
                   f"skipped: API preflight failed ({detail}) - VPN on? Rerun or use --force-scrape")
        else:
            before = os.path.getmtime(os.path.join(ROOT, "sportsbet", "sportsbet_win_draw_win_odds.csv"))
            step("Sportsbet scrape", PY + [os.path.join(ROOT, "sportsbet.py")], fatal=False)
            after = os.path.getmtime(os.path.join(ROOT, "sportsbet", "sportsbet_win_draw_win_odds.csv"))
            if after == before:
                report("Sportsbet freshness", False, "odds files were NOT refreshed - check the scrape output")

    step("Stage FFS lineups + team news", PY + [os.path.join(ROOT, "starting_lineups.py")], fatal=False)

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

    pipeline = PY + ["-m", "fpl_pipeline.run"]
    if gw:
        pipeline += ["--gw", gw]
        if "--force-archive" in argv:
            pipeline += ["--force-archive"]
    else:
        print("\nNo --gw given: projections rebuild but archives are not recorded.")
    step("Rebuild projections", pipeline)

    optimiser_py = [VENV_PY] if os.path.exists(VENV_PY) else [sys.executable]
    step("Transfer optimiser", optimiser_py + ["-X", "utf8", os.path.join(ROOT, "optimisation_gameweek.py")],
         fatal=False)
    summary_and_exit(0)


if __name__ == "__main__":
    if "--resume" in sys.argv:
        phase2(sys.argv)
    else:
        phase1(sys.argv)
