"""GW10 'Spectacular' — +1 per shot on target from OUTSIDE the box, +5 per goal from outside the
box. One player per club.

    extra = start × (1 × outside_box_shots_on_target_per90 + 5 × outside_box_goals_per90)

added to normal XP (captain doubles the boosted total). Per-90 rates come from Understat shot
data (build_longrange.py), blended THIS season with LAST season at half weight, by minutes:
    rate = (this-season count + ½ × last-season count) × 90 ÷ (this mins + ½ × last mins + SHRINK)
Goals use outside-box xG rather than actual goals: long-range goals are rare and swingy, and xG is
the steadier measure of how often a player scores from distance. SHRINK pseudo-minutes pull small
samples toward 0. Players with no Understat shots default to 0.
"""
import argparse
import os
import unicodedata

import pandas as pd

import challenge_core as cc

LONGRANGE_CSV = os.path.join(cc.INPUTS, "fpl_challenge_longrange.csv")
SEASON, LAST_SEASON = 2026, 2025
LAST_WEIGHT = 0.5
SHRINK = 270.0
SOT_PTS, GOAL_PTS = 1.0, 5.0

# Understat name -> FPL master name, for ones the matcher can't resolve on its own
ALIASES = {
    "Joshua King": "Josh King",
    "Emersonn": "Emersonn Correia da Silva",
    "Mohamed Ali-Cho": "Mohamed-Ali Cho",
    "Guemissongui Ouattara": "Abdoul Ouattara",
    "Yehor Yarmolyuk": "Yehor Yarmoliuk",
    "Chema Andres": "José María Andrés Baixauli",
    "Emile Smith-Rowe": "Emile Smith Rowe",
    "Amad Diallo Traore": "Amad Diallo",
    "Jaden Philogene-Bidace": "Jaden Philogene",
    "Murillo": "Murillo dos Santos",
}


def _norm(s):
    s = str(s).replace("ı", "i").replace("ø", "o").replace("Ø", "O").replace("ß", "ss")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def build_rates(df):
    """Per-player (outside-box shots on target /90, outside-box xG /90), blended across two
    seasons, aligned to df.index."""
    if not os.path.exists(LONGRANGE_CSV):
        cc._big_warning([f"GW10: {os.path.basename(LONGRANGE_CSV)} not found - long-range points are 0.",
                         "Run build_longrange.py first."])
        zero = pd.Series(0.0, index=df.index)
        return zero, zero, []
    lr = pd.read_csv(LONGRANGE_CSV)
    agg = {}
    for _, r in lr.iterrows():
        w = 1.0 if int(r["season"]) == SEASON else LAST_WEIGHT
        a = agg.setdefault(r["player"], [0.0, 0.0, 0.0])
        a[0] += w * r["obox_sot"]
        a[1] += w * r["obox_xg"]
        a[2] += w * r["minutes"]
    rates = {name: (90.0 * s / (m + SHRINK), 90.0 * x / (m + SHRINK)) for name, (s, x, m) in agg.items()}

    # map Understat names to master names: alias, exact (accent-insensitive), then unique surname
    by_key = dict(zip(df["Player Name"].map(_norm), df["Player Name"]))
    surname = {}
    for nm in df["Player Name"]:
        surname.setdefault(_norm(nm).split()[-1], []).append(nm)
    mapped, missed = {}, []
    for und, rate in rates.items():
        canon = ALIASES.get(und) or by_key.get(_norm(und))
        if canon is None:
            cands = surname.get(_norm(und).split()[-1], [])
            canon = cands[0] if len(cands) == 1 else None
        if canon is None:
            if rate[0] + rate[1] > 0.05:                 # only worth reporting if he shoots from range
                missed.append(und)
            continue
        mapped[canon] = rate
    sot90 = df["Player Name"].map(lambda n: mapped.get(n, (0.0, 0.0))[0])
    xg90 = df["Player Name"].map(lambda n: mapped.get(n, (0.0, 0.0))[1])
    return sot90, xg90, missed


def main(confirmed_not_starting=None, confirmed_starting=None,
         current_team=None, captain=None, now=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--confirmed-not-starting", nargs="*", default=[], metavar="NAME",
                    dest="cns", help="players confirmed benched/out — removed from the pool")
    ap.add_argument("--confirmed-starting", nargs="*", default=[], metavar="NAME",
                    dest="cs", help="players confirmed in the XI — start set to 1.0")
    args = ap.parse_args()

    df = cc.load_players(gw=10)   # snapshots/gw10.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []) + list(args.cns))
    df = cc.confirm_starting(df, list(confirmed_starting or []) + list(args.cs))
    cc.set_live(df, current_team, captain, now)   # mid-week: lock what's already been played

    sot90, xg90, missed = build_rates(df)
    if missed:
        cc._big_warning([f"GW10: {len(missed)} long-range shooter(s) not matched to FPL names:",
                         ", ".join(missed[:12]) + (" ..." if len(missed) > 12 else ""),
                         "Add them to ALIASES in gw10.py."])
    start = df["F1 Start"].fillna(0.0)
    extra = start * (SOT_PTS * sot90 + GOAL_PTS * xg90)

    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.10

    matched = int((sot90 + xg90 > 0).sum())
    title = ["FPL CHALLENGE  |  GW10 'Spectacular'  |  +1 per outside-box shot on target, +5 per outside-box goal",
             f"rates: Understat, this season + last season at {LAST_WEIGHT:g} weight, by minutes "
             f"({matched} players with long-range shots)"]
    cc.solve_and_report(df, args.max_per_club, title, "LONG")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    # current_team / captain: once your team is set and matches start, list your six and captain —
    # anyone whose match has kicked off is locked in, and the armband can't move onto a played match.
    main(confirmed_not_starting=['Jordan Pickford'], confirmed_starting=['David Raya', 'Ezri Konsa', 'Gabriel Magalhaes', 'Christos Tzolis'], current_team=[], captain=None)
