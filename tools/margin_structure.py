"""Can we measure Betway's player-market margin structure from its prices? NO — here is why.

    python tools/margin_structure.py betway_matches.har

MOTIVATION. The pipeline de-margins player prices with a flat constant
(config.MARGIN_PLAYER = 1.05, applied as 1/odds/1.05). That cannot be right: summed
implied probabilities across a fixture's goalscorer market come to ~3x what is physically
possible. The suspicion was favourite-longshot bias — thin margin on short prices, heavy on
long — which would mean the pipeline overstates unlikely scorers RELATIVE to likely ones,
and relative is what matters because projections use factors.

TWO ATTEMPTS, BOTH FAILED, BOTH THE SAME WAY.
  1. One margin per player, fitted with his lambda from the ladder shape. Gave k=0.14 at
     long prices — the bookmaker paying MORE than true odds. Impossible. Margin does not
     travel with the player; it travels with the PRICE, and one player's ladder spans
     several price levels.
  2. One margin per THRESHOLD, shared across players, fitted jointly with each player's
     lambda. Gave 1.071 / 0.911 / 0.824 / 0.904 / 1.648 for 1+..5+ — again below 1.0 in the
     middle, again impossible.

THE REAL PROBLEM: shots on target are OVER-DISPERSED, not Poisson. Minutes, role and game
state cluster shots, so variance exceeds the mean. Forcing a Poisson fit drives lambda up to
reach the tail and pushes the mid thresholds below prediction — exactly the sub-1.0 pattern
both attempts produced. Margin and distributional misfit are CONFOUNDED. Any method that
assumes a distribution in order to back out margin will fail the same way; do not try a
third variant of this.

WHAT DOES WORK is comparison that needs no distributional assumption: two prices for the
SAME event. The fixed-vs-ladder measurement (tools/betway.py) is exactly that, and it is
reliable — ladders carry ~25-30% more margin than the fixed markets, measured at 0.67-0.85
across goals, assists and shots.

VERDICT: leave MARGIN_PLAYER at 1.05. It is probably wrong, but the part that is wrong is
the SHAPE, and the shape is not measurable from prices alone. The honest test is calibration
against OUTCOMES once gameweeks accumulate: compare projected probabilities to what actually
happened, split by predicted probability, and see whether longshots are overstated.
"""
import collections
import json
import math
import sys

import numpy as np


def tail(lam, k):
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))


def fit_joint(players, rounds=25):
    """Fit one lam per player AND one margin k per THRESHOLD, alternating.

    The first attempt assumed a single margin per player, constant across his thresholds.
    It produced k=0.14 at long prices — the bookmaker paying MORE than true odds, which is
    impossible — and that failure is the finding: margin does not travel with the player,
    it travels with the PRICE. A player's 4+ shot price is far longer than his 1+ price, so
    his own ladder spans several margin levels.

    So: lam_i per player (his true rate), k_j per threshold (the load at that price level),
    shared across all players. Alternating least squares. k is reported RELATIVE to the 1+
    level, which needs no external anchor and answers the question directly — how much more
    is charged on a longer price than a short one.
    """
    ks = {j: 1.0 for j in range(1, 6)}
    lam = {}
    for _ in range(rounds):
        for who, lv in players.items():                       # lam | k
            best = (9e9, None)
            for l100 in range(3, 600):
                L = l100 / 100
                err = sum((ks[j] * tail(L, j) - p) ** 2 for j, p in lv.items())
                if err < best[0]:
                    best = (err, L)
            lam[who] = best[1]
        for j in range(1, 6):                                 # k | lam
            num = den = 0.0
            for who, lv in players.items():
                if j in lv and who in lam:
                    t = tail(lam[who], j)
                    num += lv[j] * t
                    den += t * t
            if den > 0:
                ks[j] = num / den
    return lam, ks


def main(path):
    sys.path.insert(0, ".")
    import tools.betway as bw
    har = json.load(open(path, encoding="utf-8"))
    ev = collections.defaultdict(lambda: collections.defaultdict(dict))
    for e in har["log"]["entries"]:
        u, b = e["request"]["url"], e["response"]["content"].get("text") or ""
        if "MarketGroupNamesAndMarketsForEvent" in u and b.startswith("{"):
            q = dict(p.split("=", 1) for p in u.split("?")[1].split("&") if "=" in p)
            for m, sel, sbv, odds in bw.selections(json.loads(b)):
                ev[q["eventId"]][m][(sel, sbv)] = odds

    players = {}
    for eid, bm in ev.items():
        levels = collections.defaultdict(dict)
        for lvl in range(1, 6):                       # fixed markets
            for (s, _), o in bm.get(f"Player {lvl}+ Shots On Target", {}).items():
                levels[bw.player_name(s)][lvl] = 1 / o
        for (s, _), o in bm.get(bw.SOT_LADDER, {}).items():   # ladder fills the tail
            nm, _, t = s.rpartition(" ")
            if t.endswith("+") and t[:-1].isdigit():
                levels[bw.player_name(nm)].setdefault(int(t[:-1]), 1 / o)
        for who, lv in levels.items():
            if len(lv) >= 3:
                players[(eid, who)] = lv

    fitted_lam, ks = fit_joint(players)
    base = ks[1]
    print()
    print(f"{len(players)} players, {len(ev)} fixtures - one lambda each, one margin per threshold")
    print()
    print(f"{'threshold':<12}{'margin k':>10}{'rel. to 1+':>13}   typical price")
    for j in sorted(ks):
        px = [1 / v[j] for v in players.values() if j in v]
        if len(px) >= 5:
            print(f"  {j}+{'':<9}{ks[j]:>10.3f}{ks[j]/base:>12.2f}x   ~{np.median(px):.1f}")
    print()
    print("Rising multiple = MORE margin on longer prices (favourite-longshot bias).")
    print("config.MARGIN_PLAYER = 1.05 applies the SAME haircut at every price.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "betway_matches.har")
