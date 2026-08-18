"""Team-gameweek table of REAL odds, gathered forward across gameweeks.

    python tools/team_gw_odds_table.py [--fixtures 10]

One row per (Season, Gameweek, Team) — the player archive repeats every team-level odds
column across that team's players, so dropping duplicates recovers the team view
losslessly.

Two different as-of dates, deliberately:

  * WHO and WHAT HAPPENED come from the future gameweek. F{k} Win / Opponent Win /
    Venue / Opponent are read from that team's **gameweek M+k-1 row**, taking that
    row's own F1 (real, odds-derived) values. F2-F8 *Pred* columns are never used.

  * WHAT WAS KNOWN comes from gameweek M. The team's own title/relegation/top-6 and
    **every fixture's opponent outrights** are as at M, not as at the fixture itself.
    On Arsenal's GW21 row, F10 names the team Arsenal met in GW30, but prices that
    opponent with its GW21 outrights — what the market thought of them back at M.

That split is the point: features are what the market knew at M, targets are the real
match odds that materialised k-1 gameweeks later.

The F2-F8 *Pred* columns are deliberately never used. Every number here is what the
market actually said at the time, so the table is a clean target for "given the season
odds at M, what were the real match odds k-1 gameweeks later" — the question behind the
odds-persistence model (task #19) and the reliability half of the optimiser's fixture
weights.

Blank means that gameweek was never archived (the 2025-26 archive covers GW16-27 and 29,
so GW28 and anything past 29 is empty, and late gameweeks run out of future rows).
Note F{k} indexes GAMEWEEK offset, not fixture number — a blank gameweek yields a blank
cell rather than shifting the sequence.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

ARCHIVE = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
KEY = ["Season", "Gameweek", "Team"]

# The team's own season odds, as at its own gameweek
CONTEXT = {"Title": "Title", "Relegation": "Relegation", "Top 6": "Top 6"}

# Gathered forward across gameweeks. Venue and opponent identity belong to the fixture,
# not the row, so they travel with the win pair.
WIN = {"F1 Win": "Win", "F1 Opponent Win": "Opponent Win", "F1 Venue": "Venue",
       "F1 Opponent": "Opponent"}

# Outrights are per (season, gameweek, team) and get looked up at the ROW's gameweek
OUTRIGHTS = ["Title", "Relegation", "Top 6"]

# Plausible implied-draw band for a Premier League match. The floor is loose enough to
# admit genuine heavy mismatches; the ceiling is well above any real 3-way book.
DRAW_MIN, DRAW_MAX = 0.12, 0.38


def team_view():
    """One row per team-gameweek, carrying only that gameweek's real values."""
    keep = set(KEY) | set(CONTEXT) | set(WIN)
    archive = pd.read_csv(ARCHIVE, low_memory=False, usecols=lambda c: c in keep)
    team = archive.drop_duplicates(subset=KEY).rename(columns={**CONTEXT, **WIN})
    team["Gameweek"] = pd.to_numeric(team["Gameweek"], errors="coerce").astype("Int64")
    return (team.dropna(subset=["Gameweek"])
                .sort_values(["Season", "Team", "Gameweek"])
                .reset_index(drop=True))


def gather_forward(team, n_fixtures):
    """Build the two-as-of-date table (see module docstring)."""
    win_cols = list(WIN.values())
    out = team.drop(columns=win_cols)

    # (season, gameweek, team) -> that team's outrights, for pricing opponents at M
    book = team.set_index(["Season", "Gameweek", "Team"])[OUTRIGHTS]

    for k in range(1, n_fixtures + 1):
        src = team[KEY + win_cols].copy()
        # a row at gameweek G supplies F{k} for the row at G-(k-1)
        src["Gameweek"] = src["Gameweek"] - (k - 1)
        src = src.rename(columns={c: f"F{k} {c}" for c in win_cols})
        out = out.merge(src, on=KEY, how="left")

        # Price that fixture's opponent using the ROW's gameweek, not the fixture's
        idx = pd.MultiIndex.from_arrays(
            [out["Season"], out["Gameweek"], out[f"F{k} Opponent"]])
        found = book.reindex(idx)
        for col in OUTRIGHTS:
            out[f"F{k} Opponent {col}"] = found[col].values

    # F1 is the row's own gameweek, so its block adds nothing the row does not already
    # carry: the team's own outrights are the unprefixed columns, and the F1 opponent
    # block is the sparsest in the table. Dropped so every F{k} column is genuinely a
    # FUTURE fixture.
    return out.drop(columns=[c for c in out.columns if c.startswith("F1 ")])


def drop_corrupt(team, verbose=True):
    """Remove rows the archive got wrong, BEFORE gathering forward.

    Order matters: a clean GW24 row reads its F2 from GW25, so filtering only the output
    rows would leave corrupt values embedded in clean rows' fixture columns. Dropping at
    source means anything that referenced bad data comes back blank instead.

    Not dropped: GW16-20's missing opponent names. Missing is not wrong — the win
    probabilities there are complete and correct, and a blank is honest.
    """
    n0 = len(team)
    dropped = []

    # A stale fixture window corrupts a whole gameweek at once (build_fixtures.py was
    # not re-run). One or two teams repeating is possible via rearranged fixtures, so
    # only a systemic repeat condemns the gameweek.
    s = team.sort_values(["Team", "Gameweek"])
    repeat = s[(s["Opponent"] == s.groupby("Team")["Opponent"].shift()) & s["Opponent"].notna()]
    counts = repeat.groupby("Gameweek").size()
    n_teams = team.groupby("Gameweek")["Team"].nunique()
    stale = [gw for gw, c in counts.items() if c >= 0.5 * n_teams.get(gw, 20)]
    if stale:
        team = team[~team["Gameweek"].isin(stale)]
        dropped.append(f"GW{', GW'.join(str(int(g)) for g in stale)} entirely "
                       f"(stale fixture window: opponents/venues belong to the previous gameweek)")

    # Win + Opponent Win must leave a sane share for the draw. Bookmakers price PL draws
    # around 0.20-0.28, compressing to ~0.12 only for extreme mismatches (Arsenal 0.88 v
    # Wolves is real, not a defect). A gameweek where a QUARTER of matches fall outside
    # that band has had its odds scrambled across fixtures wholesale, not by accident:
    # 2025-26 GW25 and GW29 ran at 62% and 57% violations with implied draws from 0.03
    # to 0.67, against 0-10% and a tight 0.21-0.24 mean everywhere else.
    draw = 1 - team["Win"] - team["Opponent Win"]
    off = (draw < DRAW_MIN) | (draw > DRAW_MAX)
    rate = off.groupby(team["Gameweek"]).mean()
    scrambled = [gw for gw, r in rate.items() if r > 0.25]
    if scrambled:
        detail = ", ".join(f"GW{int(g)} ({rate[g]:.0%} of matches)" for g in scrambled)
        team = team[~team["Gameweek"].isin(scrambled)]
        dropped.append(f"{detail} entirely (implied draw out of range — odds scrambled "
                       f"across fixtures)")

    # Residual single-row defects in otherwise healthy gameweeks
    draw = 1 - team["Win"] - team["Opponent Win"]
    impossible = draw < 0.0
    if impossible.any():
        team = team[~impossible]
        dropped.append(f"{int(impossible.sum())} further rows with a negative implied draw")

    if verbose:
        if dropped:
            print(f"removed corrupt data: {n0} -> {len(team)} team-gameweeks")
            for d in dropped:
                print(f"  - {d}")
        else:
            print("no corrupt data found")
    return team.reset_index(drop=True)


def to_long(table, n_fixtures, keep_empty=False):
    """One row per (team, gameweek, future fixture) instead of one per team-gameweek.

    `Fixture` is the fixture number (2..n) and `Target Gameweek` is the gameweek it was
    played in — the same arithmetic the wide columns encode, made explicit. Rows with no
    fixture (that gameweek was never archived, or was dropped as corrupt) are removed
    unless keep_empty, since a long table is one row per observation.
    """
    id_cols = ["Season", "Gameweek", "Team", "Title", "Relegation", "Top 6"]
    per_fixture = ["Win", "Opponent Win", "Venue", "Opponent",
                   "Opponent Title", "Opponent Relegation", "Opponent Top 6"]

    frames = []
    for k in range(2, n_fixtures + 1):
        cols = {f"F{k} {c}": c for c in per_fixture if f"F{k} {c}" in table.columns}
        if not cols:
            continue
        block = table[id_cols + list(cols)].rename(columns=cols)
        block.insert(len(id_cols), "Fixture", k)
        block.insert(len(id_cols) + 1, "Target Gameweek", block["Gameweek"] + k - 1)
        frames.append(block)

    long = pd.concat(frames, ignore_index=True)
    if not keep_empty:
        long = long[long["Win"].notna()]
    return long.sort_values(["Season", "Team", "Gameweek", "Fixture"]).reset_index(drop=True)


def data_quality(team):
    """Report archive defects that would silently corrupt anything fitted on this table.

    Both checks below found real problems in the 2025-26 archive, so they run every time
    rather than living in a notebook someone forgets to open.
    """
    problems = []

    # 1. A stale fixture window: run.py --gw N was run without build_fixtures.py --gw N,
    #    so the archive recorded the PREVIOUS gameweek's opponents against new odds.
    s = team.sort_values(["Team", "Gameweek"])
    repeat = s[(s["Opponent"] == s.groupby("Team")["Opponent"].shift())
               & s["Opponent"].notna()]
    for gw, n in repeat.groupby("Gameweek").size().items():
        problems.append(
            f"GW{int(gw)}: {n} teams repeat GW{int(gw) - 1}'s opponent"
            + (" — ALL of them, so the fixture window was stale for the whole gameweek"
               if n >= 20 else ""))

    # 2. The implied draw must be plausible. A LOW draw is legitimate when one side is a
    #    heavy favourite (Arsenal 0.88 v Wolves genuinely squeezes the draw to ~0.08), so
    #    only flag it when no such mismatch explains it. A HIGH draw never has an
    #    innocent explanation.
    draw = 1 - team["Win"] - team["Opponent Win"]
    favourite = team[["Win", "Opponent Win"]].max(axis=1)
    suspect = (draw > DRAW_MAX) | ((draw < DRAW_MIN) & (favourite < 0.75))
    bad = team[suspect]
    if len(bad):
        by_gw = ", ".join(f"GW{int(g)} ({n})" for g, n in bad.groupby("Gameweek").size().items())
        problems.append(f"{len(bad)} rows imply an unexplained draw outside "
                        f"{DRAW_MIN}-{DRAW_MAX}: {by_gw}")
    mismatch = (draw < DRAW_MIN) & (favourite >= 0.75)
    if mismatch.any():
        print(f"  (note: {int(mismatch.sum())} rows have draw < {DRAW_MIN} explained by a "
              f"heavy favourite — kept, they are real)")

    print("data quality" + (":" if problems else ": no defects detected"))
    for p in problems:
        print(f"  ! {p}")
    if problems:
        print("  (rows are still written — filter them out before fitting)")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=int, default=10)
    ap.add_argument("--keep-corrupt", action="store_true",
                    help="keep rows the archive got wrong (default drops them)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="long form: keep rows for fixtures that were never archived")
    args = ap.parse_args()

    team = team_view()
    if not args.keep_corrupt:
        team = drop_corrupt(team)
    table = gather_forward(team, args.fixtures)
    path = os.path.join(config.OUTPUTS_DIR, "team_gw_real_odds.csv")
    table.to_csv(path, index=False)

    gws = team["Gameweek"]
    print(f"{len(table)} team-gameweeks | seasons {sorted(team['Season'].unique())} "
          f"| GW{gws.min()}-{gws.max()} | {len(table.columns)} columns -> {path}\n")
    long = to_long(table, args.fixtures, keep_empty=args.keep_empty)
    long_path = os.path.join(config.OUTPUTS_DIR, "team_gw_real_odds_long.csv")
    long.to_csv(long_path, index=False)
    print(f"{len(long)} rows x {len(long.columns)} columns (long) -> {long_path}\n")

    print("rows per fixture number (blank = that gameweek not archived):")
    for k in range(1, args.fixtures + 1):
        col = f"F{k} Win"
        if col in table.columns:
            print(f"  F{k:<3} {int(table[col].notna().sum()):>4} / {len(table)}")
    print()
    data_quality(team)
