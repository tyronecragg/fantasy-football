"""Cross-check curated start probabilities against FPL's own availability flags.

    python tools/injury_check.py

FPL's `status` / `chance_of_playing_next_round` answer "can he play at all?", which is
NOT our start probability ("will he be in the XI?"). They are used here only as a
CEILING on our belief:

    status a          -> no constraint (says nothing about starting)
    status d, 75%     -> start_prob must be <= 0.75 (he might not even be available)
    status i/u/s, 0%  -> start_prob must be 0

So a 100%-available bench player can still be graded 0.15, and a 75% doubt who would
walk into the XI when fit tops out at 0.75. This reports violations of that ceiling.
Nothing is auto-applied: transfers out (a sale FPL hasn't processed) legitimately look "fit".
"""
import os
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names  # noqa: E402
from fpl_pipeline.io_utils import read_csv_tolerant  # noqa: E402

# Distinct codepoints that NFKD leaves intact and ascii-encoding would then delete
LETTERS = {"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
           "ß": "ss", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ı": "i"}


def fold(name):
    s = str(name)
    for a, b in LETTERS.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.replace("-", " ").lower().split())


def availability():
    """Per-player FPL availability, keyed by folded full name."""
    stats = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "playerstats.csv"))
    if "gw" in stats.columns:
        stats = stats.sort_values("gw").drop_duplicates(subset="id", keep="last")
    # Same rename table the pipeline uses, so these keys match our curated names
    full = names.apply_player_names(stats["first_name"] + " " + stats["second_name"])
    stats["_fold"] = full.map(fold)
    stats["ceiling"] = stats.apply(
        lambda r: 1.0 if r["status"] == "a" else (r["chance_of_playing_next_round"] or 0) / 100.0,
        axis=1)
    return stats.set_index("_fold")[["web_name", "status", "ceiling", "news"]]


def main():
    avail = availability()
    lineups = read_csv_tolerant(os.path.join(config.INPUTS_DIR, "starting_lineups.csv"))

    over, unmatched = [], []
    for row in lineups.itertuples():
        key = fold(row.Player)
        if key not in avail.index:
            unmatched.append(row.Player)
            continue
        a = avail.loc[key]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        # F1 is the gameweek FPL's "next round" flag describes
        if row.F1 > a["ceiling"] + 1e-9:
            over.append((row.Player, row.Team, row.F1, a["status"], a["ceiling"], a["news"]))

    print("=" * 78)
    print("START PROBABILITIES ABOVE FPL'S AVAILABILITY CEILING (F1)")
    print("=" * 78)
    if over:
        for player, team, prob, status, ceiling, news in sorted(over, key=lambda r: r[2] - r[4], reverse=True):
            print(f"  {player:<24} {team:<14} ours {prob:.2f} > max {ceiling:.2f} "
                  f"(status {status}) {str(news or '')[:42]}")
    else:
        print("  none - every graded player is within what FPL says is possible")

    if unmatched:
        print(f"\n{len(unmatched)} lineup players not matched in FPL data "
              f"(check name mappings): {', '.join(unmatched[:8])}"
              f"{' ...' if len(unmatched) > 8 else ''}")


if __name__ == "__main__":
    main()
