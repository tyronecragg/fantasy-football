"""GW6 'Matchwinner' — winning goals are worth an extra 10 points. One player per club.

A winning goal needs the team to win AND this player to score the decisive goal, so we
add each player's expected winning-goal points:

    extra = 10 × P(team wins) × goal_share(player)

goal_share = the player's expected goals ÷ the team's expected goals (from the goalscorer
odds). Shares within a team sum to 1, so exactly one winning goal is spread across the
winning side. P(team wins) is the win-odds column. Captain then doubles the boosted total.

    python gw6.py
"""

import argparse

import challenge_core as cc


def main(confirmed_not_starting=None, confirmed_starting=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--confirmed-not-starting", nargs="*", default=[], metavar="NAME",
                    dest="cns", help="players confirmed benched/out — removed from the pool")
    ap.add_argument("--confirmed-starting", nargs="*", default=[], metavar="NAME",
                    dest="cs", help="players confirmed in the XI — start set to 1.0")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []) + list(args.cns))
    df = cc.confirm_starting(df, list(confirmed_starting or []) + list(args.cs))

    eg = cc.expected_goals(df)
    team_eg = df.assign(_eg=eg).groupby("Team")["_eg"].transform("sum")
    share = (eg / team_eg).fillna(0.0)
    pwin = df["F1 Win"].fillna(0.0) if "F1 Win" in df.columns else 0.0
    extra = 10.0 * pwin * share                      # expected winning-goal points

    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]                    # captain doubles the boosted total
    df["boosted"] = extra > 0.10

    title = ["FPL CHALLENGE  |  GW6 'Matchwinner'  |  winning goals worth an extra 10 pts",
             "boost = 10 × P(team wins) × player's share of team goals"]
    cc.solve_and_report(df, args.max_per_club, title, "MW +10")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    main(confirmed_not_starting=[], confirmed_starting=['Cole Palmer'])
