# -*- coding: utf-8 -*-
"""Weekly chip-value log — Bench Boost, Triple Captain, Wildcard, and Free Hit, one row per gameweek.

    env/Scripts/python tools/chip_history.py [--gw N] [--season S] [--free-transfers 1] [--bank 0.0] [--budget B] [--radar]

--radar additionally prints a FORWARD chip radar: each of F1..F8 valued as a standalone, full-weight
one-off (Bench Boost / Triple Captain / Free Hit), so a spike anywhere in the 8-fixture window is
visible in advance — not only when it becomes the next fixture.

Appends to inputs/chip_history.csv so a SPIKE in any column flags a week a chip may be worth
playing (a blank/double GW, a dream captain fixture, a bench full of playing assets). Re-running
the same gameweek REPLACES its row (upsert by Season+Gameweek).

  BENCH BOOST value    = F1 XP the 4 bench players would add if they counted (current squad, best
                         XI fielded, other 4 = bench). Single-GW chip → F1 only. Cheap.
  TRIPLE CAPTAIN value = your captain's F1 XP — the EXTRA 1x you gain going from 2x to 3x (on a
                         double gameweek the master's F1 already carries both matches, so it
                         spikes on its own). Single-GW chip → F1 only. Cheap.

  Both rebuild deltas are measured against a BASELINE = the NORMAL team you'd field: the best team
  reachable with your `--free-transfers` free transfers (default 1) on the 8-fixture objective, NOT
  your raw current squad and NOT a team myopically built for one fixture — so they show the chip's
  value ON TOP of what you'd do anyway. Captaincy is included on both sides.

  WILDCARD differential = best team with 15 transfers (a full rebuild) MINUS the baseline, on the
                         optimiser's 8-FIXTURE objective (`Σ weight_f · Ff XP`, ownership×reliability
                         weights normalised to F1=1.0) + captaincy — a permanent squad, judged over
                         the horizon.
  FREE HIT differential = best team with 15 transfers MINUS the baseline, on F1 ONLY + captaincy —
                         a one-week team, so a single gameweek.

Both deltas are budget-constrained ILPs; they need the venv (PuLP) and are blank without it. Pass
--gw to match your archive gameweek.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, history  # noqa: E402

MASTER = os.path.join(ROOT, "outputs", "players_master.csv")
GW_TEAMS = os.path.join(ROOT, "inputs", "gw_teams.csv")
HIST = os.path.join(ROOT, "inputs", "chip_history.csv")
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FIX_COLS = [f"F{k} XP" for k in range(1, 9)]
# Optimiser's 8-fixture weights (ownership × reliability, F1-normalised; reliability measured in
# points space 2026-09-02, see optimisation.RELIABILITY_WEIGHTS) — replicated from
# optimisation.py rather than imported, because importing that module fires its top-level run.
_OWN = (1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558)
_REL = (1.0, 0.850, 0.710, 0.630, 0.550, 0.470, 0.400, 0.330)   # points-measured 2026-09-02
HORIZON_W = [(o * r) / (_OWN[0] * _REL[0]) for o, r in zip(_OWN, _REL)]


def current_squad_names(target_gw):
    """The 15 names you carried INTO `target_gw` — i.e. your GW(target_gw-1) squad. That is the
    baseline the chip deltas are measured against; your `--free-transfers` are then applied on top of
    it to reach the normal team. Reading the target GW's own (already-transferred) column instead
    would double-count the transfer you already spent. GW1 has no prior week, so it uses the GW1
    initial squad. Falls back to the most recent populated column strictly before target_gw."""
    df = pd.read_csv(GW_TEAMS)

    def col_names(c):
        if c not in df.columns:
            return []
        return [n for n in df[c].dropna().astype(str).str.strip() if n]

    for g in range(target_gw - 1, 0, -1):          # most recent populated week BEFORE target_gw
        names = col_names(f"GW{g}")
        if names:
            return names
    names = col_names(f"GW{target_gw}")            # GW1 case: no prior week, use the initial squad
    if names:
        return names
    raise SystemExit(f"no populated GW column at or before GW{target_gw} in gw_teams.csv")


def best_xi(sub, col):
    """Max-`col` legal XI from a squad frame (index preserved). Returns (xi_value, set(xi_index))."""
    by = {p: sub[sub.Position == p].sort_values(col, ascending=False) for p in XI_MIN}
    xi = []
    for p in XI_MIN:
        xi += list(by[p].index[:XI_MIN[p]])
    rest = pd.concat([by["DEF"].iloc[XI_MIN["DEF"]:XI_MAX["DEF"]],
                      by["MID"].iloc[XI_MIN["MID"]:XI_MAX["MID"]],
                      by["FWD"].iloc[XI_MIN["FWD"]:XI_MAX["FWD"]]]).sort_values(col, ascending=False)
    xi += list(rest.index[:11 - len(xi)])
    return float(sub.loc[xi, col].sum()), set(xi)


def captain_horizon(sub, xi_idx):
    """Horizon captaincy value for a fixed XI: Σ_f weight_f · max over the XI of that fixture's XP
    (you captain the best available starter each week). The +1x a captain adds over 8 fixtures."""
    xi = sub.loc[list(xi_idx)]
    return float(sum(w * xi[c].max() for w, c in zip(HORIZON_W, FIX_COLS)))


def best_squad(pool, budget, col, current_idx=None, max_transfers=15, want="xi",
               bench_slots=None, gk_bench=0.0):
    """Indices of the best legal XI maximising the sum of `col` over an 11-man XI inside a 15-man
    squad, within `budget` and <=3 players per team, changing AT MOST `max_transfers` players from
    `current_idx` (None / 15 = a free rebuild). `want`: "xi" -> the 11 starters (default),
    "squad" -> the 15-man squad, "both" -> (squad_set, xi_set). Returns None if PuLP is missing /
    the problem is infeasible. Captaincy is added post-hoc by the *_total helpers.

    `bench_slots` (s1, s2, s3) additionally values the 4 bench by sub order (best `col` = slot 1,
    sum-of-top-k encoding) + `gk_bench` for the backup keeper, so the rebuild shapes a real bench
    rather than fodder. None (default) = XI-only. Score the result with `bench_value` at the same
    weights so selection and scoring agree."""
    try:
        import pulp
    except ImportError:
        return None
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    idx = list(pool.index)
    sq = {i: pulp.LpVariable(f"sq_{i}", cat="Binary") for i in idx}
    st = {i: pulp.LpVariable(f"st_{i}", cat="Binary") for i in idx}
    objective = [pool.loc[i, col] * st[i] for i in idx]                     # XI
    if bench_slots is not None:
        bn = {i: pulp.LpVariable(f"bn_{i}", cat="Binary") for i in idx}     # bench = squad AND not starting
        for i in idx:
            prob += bn[i] <= sq[i]
            prob += bn[i] <= 1 - st[i]
            prob += bn[i] >= sq[i] - st[i]
        s1, s2, s3 = bench_slots
        outfield = [i for i in idx if pool.loc[i, "Position"] != "GK"]
        objective += [pool.loc[i, col] * gk_bench * bn[i] for i in idx if pool.loc[i, "Position"] == "GK"]
        objective += [pool.loc[i, col] * s3 * bn[i] for i in outfield]      # every outfield sub earns slot-3
        top1 = {i: pulp.LpVariable(f"b1_{i}", lowBound=0, upBound=1) for i in outfield}
        top2 = {i: pulp.LpVariable(f"b2_{i}", lowBound=0, upBound=1) for i in outfield}
        for i in outfield:
            prob += top1[i] <= bn[i]
            prob += top2[i] <= bn[i]
        prob += pulp.lpSum(top1.values()) <= 1                             # best sub adds (s1-s2) more
        prob += pulp.lpSum(top2.values()) <= 2                             # best two add (s2-s3) more
        objective += [pool.loc[i, col] * (s1 - s2) * top1[i] for i in outfield]
        objective += [pool.loc[i, col] * (s2 - s3) * top2[i] for i in outfield]
    prob += pulp.lpSum(objective)
    prob += pulp.lpSum(sq.values()) == 15
    prob += pulp.lpSum(st.values()) == 11
    prob += pulp.lpSum(pool.loc[i, "Cost"] * sq[i] for i in idx) <= budget
    for i in idx:
        prob += st[i] <= sq[i]
    for pos, n in SQUAD.items():
        pidx = pool.index[pool.Position == pos]
        prob += pulp.lpSum(sq[i] for i in pidx) == n
    for pos in XI_MIN:
        pidx = pool.index[pool.Position == pos]
        prob += pulp.lpSum(st[i] for i in pidx) >= XI_MIN[pos]
        prob += pulp.lpSum(st[i] for i in pidx) <= XI_MAX[pos]
    for team in pool.Team.dropna().unique():
        tidx = pool.index[pool.Team == team]
        prob += pulp.lpSum(sq[i] for i in tidx) <= 3
    if current_idx is not None and max_transfers < 15:
        # transfers = 15 - (current players kept); cap them by keeping at least 15 - max_transfers
        prob += pulp.lpSum(sq[i] for i in current_idx) >= max(0, 15 - max_transfers)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    squad_set = {i for i in idx if sq[i].value() == 1}
    xi_set = {i for i in idx if st[i].value() == 1}
    if want == "both":
        return squad_set, xi_set
    return squad_set if want == "squad" else xi_set


def horizon_total(pool, xi_idx):
    """8-fixture weighted XI value + captaincy (best available starter doubled each fixture)."""
    return float(pool.loc[list(xi_idx), "_htotal"].sum()) + captain_horizon(pool, xi_idx)


def fk_total(pool, xi_idx, col):
    """Single-fixture XI value + captaincy (best starter of that fixture doubled), for any Ff XP."""
    xi = pool.loc[list(xi_idx), col]
    return float(xi.sum() + xi.max())


def f1_total(pool, xi_idx):
    """Single-gameweek (F1) XI value + captaincy (the best F1 starter doubled)."""
    return fk_total(pool, xi_idx, "F1 XP")


# Bench slot weights for the rebuild lenses. WILDCARD = a PERMANENT squad, so its bench matters
# like a normal team (mirrors optimisation.BENCH_SLOT_WEIGHTS). FREE HIT = a ONE-WEEK squad, so
# only the first sub's autosub chance is worth anything; subs 2/3 basically never come on.
WILDCARD_BENCH_SLOTS = (0.25, 0.05, 0.01)
WILDCARD_GK_BENCH = 0.01
FREE_HIT_BENCH_SLOTS = (0.15, 0.001, 0.001)
FREE_HIT_GK_BENCH = 0.0


def bench_value(pool, squad_idx, xi_idx, col, slots, gk_bench=0.0):
    """Weighted `col` value of the 4 bench (squad minus XI): outfield subs priced by sub order
    (best `col` = slot 1), backup GK at `gk_bench`. Mirrors optimisation.bench_points_for_fixture,
    so it scores exactly what best_squad's bench_slots selected."""
    bench = [i for i in squad_idx if i not in xi_idx]
    gks = [i for i in bench if pool.loc[i, "Position"] == "GK"]
    outfield = sorted((i for i in bench if pool.loc[i, "Position"] != "GK"),
                      key=lambda i: pool.loc[i, col], reverse=True)
    v = sum(pool.loc[i, col] * gk_bench for i in gks)
    v += sum(pool.loc[i, col] * w for i, w in zip(outfield, slots))
    return float(v)


def gated_best_squad(pool, budget, current_idx, ft, min_f1_gain, min_total_gain, min_remaining_gain,
                     bench_slots, gk_bench):
    """The REALISTIC baseline: best team reachable with `ft` transfers, but the transfer(s) are kept
    ONLY if they clear the min-gain gate over HOLDING (0 transfers) - the F1 XI (incl captain) improves
    by >= min_f1_gain AND the F2-onward (tail) weighted XI+captain improves by >= min_remaining_gain AND
    the weighted 8-fixture total (incl bench) improves by >= min_total_gain; else the held squad is
    returned. Mirrors the optimiser's transfer gate, so chip deltas are measured against the team you'd
    ACTUALLY field rather than one that made a marginal (or rent-shaped) transfer. `bool` second return
    = whether the transfer was kept. Returns (squad_set, xi_set, kept) or None."""
    move = best_squad(pool, budget, "_htotal", current_idx, ft, want="both",
                      bench_slots=bench_slots, gk_bench=gk_bench)
    if move is None:
        return None
    if not (min_f1_gain > 0 or min_total_gain > 0 or min_remaining_gain > 0) or ft <= 0:
        return (*move, True)
    hold = best_squad(pool, budget, "_htotal", current_idx, 0, want="both",
                      bench_slots=bench_slots, gk_bench=gk_bench)
    if hold is None:
        return (*move, True)
    m15, mxi = move
    h15, hxi = hold
    tot_gain = ((horizon_total(pool, mxi) + bench_value(pool, m15, mxi, "_htotal", bench_slots, gk_bench))
                - (horizon_total(pool, hxi) + bench_value(pool, h15, hxi, "_htotal", bench_slots, gk_bench)))
    f1_move, f1_hold = f1_total(pool, mxi), f1_total(pool, hxi)
    f1_gain = f1_move - f1_hold
    rem_gain = (horizon_total(pool, mxi) - f1_move) - (horizon_total(pool, hxi) - f1_hold)  # weighted tail
    if tot_gain >= min_total_gain and f1_gain >= min_f1_gain and rem_gain >= min_remaining_gain:
        return (*move, True)
    return (*hold, False)


MAX_FT = 5   # FPL caps banked free transfers at 5


def _obj_value(m, squad_set, xi_set, col, slots, gk):
    """Value of a squad on `col`: starting-XI sum + weighted bench (the best_squad objective)."""
    return float(m.loc[list(xi_set), col].sum()) + bench_value(m, squad_set, xi_set, col, slots, gk)


def _hz_value(m, squad_set, xi_set, hw_col, fwd_cols, wts, slots, gk):
    """Weighted forward-horizon value of a squad: XI sum on the horizon column + per-fixture captaincy
    (best starter each week, weighted) + weighted bench. Mirrors horizon_total+bench for any window."""
    xi = list(xi_set)
    cap = float(sum(w * m.loc[xi, c].max() for w, c in zip(wts, fwd_cols)))
    return float(m.loc[xi, hw_col].sum()) + cap + bench_value(m, squad_set, xi_set, hw_col, slots, gk)


def simulate_weekly_squads(m, budget, current_idx, ft, min_f1_gain, min_remaining_gain, min_total_gain,
                           bench_slots, gk_bench):
    """Play the weekly free-transfer drip FORWARD: each fixture-week make the best transfer, but keep it
    ONLY if it clears the SAME 3-gate the optimiser/`gated_best_squad` uses (measured over HOLDING),
    with week w as that week's "F1": the F_w XI+captain improves >= min_f1_gain AND the weighted tail
    (F_{w+1}..F_{w+7}) improves >= min_remaining_gain AND the weighted window total (incl bench) improves
    >= min_total_gain; else BANK (the FT carries, capped at MAX_FT). The squad EVOLVES week to week.
    Returns ({w: (squad_set, xi_set)}, {w: ft_available}) for w=1..HORIZON = the realistic team you'd
    HOLD each week under YOUR real transfer discipline (so the GW-w baseline matches `gated_best_squad`
    for w=1). Greedy/myopic per week, as the weekly game forces."""
    n, W = config.HORIZON, HORIZON_W
    fwd_of = {}
    for w in range(1, n + 1):                                    # rolling weighted horizon from each week
        cols = [f"F{w + j} XP" for j in range(len(W)) if w + j <= n and f"F{w + j} XP" in m.columns]
        m[f"_hw{w}"] = sum(wt * m[c] for wt, c in zip(W, cols))
        fwd_of[w] = cols
    gated = min_f1_gain > 0 or min_remaining_gain > 0 or min_total_gain > 0
    evolved, ft_avail = {}, {}
    idx, avail = list(current_idx), float(ft)
    for w in range(1, n + 1):
        a = min(int(round(avail)), MAX_FT)
        col, fwd = f"_hw{w}", fwd_of[w]
        wts = W[:len(fwd)]
        hold = best_squad(m, budget, col, idx, 0, want="both", bench_slots=bench_slots, gk_bench=gk_bench)
        if hold is None:                                        # PuLP missing / infeasible -> freeze
            hold = (set(idx), set(list(idx)[:11]))
        chosen, used = hold, 0
        if a > 0:
            move = best_squad(m, budget, col, idx, a, want="both", bench_slots=bench_slots, gk_bench=gk_bench)

            def _gains(sq, xi):                                 # (F_w XI+cap, weighted tail, weighted total incl bench)
                xl = list(xi)
                f1v = fk_total(m, xi, f"F{w} XP")
                xicap = float(m.loc[xl, col].sum()) + sum(wt * m.loc[xl, c].max() for wt, c in zip(wts, fwd))
                totv = xicap + bench_value(m, sq, xi, col, bench_slots, gk_bench)
                return f1v, xicap - f1v, totv

            if move is not None:
                if not gated:
                    chosen, used = move, 15 - len(set(move[0]) & set(idx))
                else:
                    mf1, mrem, mtot = _gains(*move)
                    hf1, hrem, htot = _gains(*hold)
                    if (mtot - htot >= min_total_gain and mf1 - hf1 >= min_f1_gain
                            and mrem - hrem >= min_remaining_gain):
                        chosen, used = move, 15 - len(set(move[0]) & set(idx))
        evolved[w], ft_avail[w] = chosen, a
        idx = list(chosen[0])
        avail = min(a - used + 1, MAX_FT)                       # spent `used`, gain +1 next week
    return evolved, ft_avail


def chip_radar(m, squad, current_idx, budget, ft, min_f1_gain=0.0, min_total_gain=0.0, min_remaining_gain=0.0):
    """Forward radar: treat EACH of F1..F{HORIZON} as a standalone, full-weight one-off and value a
    Bench Boost / Triple Captain / Free Hit there — so a chip spike anywhere in the horizon is visible
    in advance, not only when it becomes F1. The baseline is the team you'd REALISTICALLY HOLD at each
    fixture, from `simulate_weekly_squads` (play the weekly FT drip forward, gated by min_total_gain):
    Bench Boost = that week's 4 subs, Triple Captain = that week's best starter, Free Hit = the
    per-fixture 15-transfer rebuild MINUS that week's evolved squad, Wildcard = the PERMANENT rebuild
    over that week's forward 8-fixture window MINUS the evolved squad ("worth wildcarding here, GIVEN
    I keep making regular transfers?" — so it decays as the drip fixes the squad). So far-week Free Hit
    and Wildcard deltas shrink honestly (your evolved team has already transferred toward those
    fixtures). The `(sim FTs)` column
    is the free transfers available that week in the simulation (spent/banked). Note: Ff slots are
    per-FIXTURE (not calendar), so double/blank GWs map only approximately."""
    # Simulate the weekly FT drip: the squad you'd actually HOLD at each fixture, having made your
    # best gated transfer (or banked) every week. Bench Boost / Triple Captain / the Free Hit baseline
    # all read THAT week's evolved squad, so far-week values reflect a team that's transferred toward
    # those fixtures - not a one-shot bulk rebuild of the current squad.
    evolved, ft_avail = simulate_weekly_squads(m, budget, current_idx, ft, min_f1_gain,
                                               min_remaining_gain, min_total_gain,
                                               WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)
    rows = []
    for k in range(1, config.HORIZON + 1):
        col = f"F{k} XP"
        ev15 = evolved[k][0]
        ev = m.loc[list(ev15)]
        _, bxi = best_xi(ev, col)                               # the XI you'd field that week
        bb = float(ev.loc[~ev.index.isin(bxi), col].sum())      # Bench Boost = the 4 subs that week
        tc = float(ev.loc[list(bxi), col].max())                # Triple Captain = best starter that week
        fh_k = best_squad(m, budget, col, current_idx, 15, want="both",     # one-week rebuild (thin bench)
                          bench_slots=FREE_HIT_BENCH_SLOTS, gk_bench=FREE_HIT_GK_BENCH)
        if fh_k:                                                # FH delta = one-week best team - your evolved team
            f15, fxi = fh_k
            fh = round((fk_total(m, fxi, col) + bench_value(m, f15, fxi, col, FREE_HIT_BENCH_SLOTS, FREE_HIT_GK_BENCH))
                       - (fk_total(m, bxi, col) + bench_value(m, ev15, bxi, col, FREE_HIT_BENCH_SLOTS, FREE_HIT_GK_BENCH)), 2)
        else:
            fh = None
        # WILDCARD delta at week k: PERMANENT rebuild over the forward 8-fixture window (F_k..F_{k+7},
        # `_hw{k}` from the sim) MINUS your evolved squad held over it. "Is it worth wildcarding here,
        # given I've been making regular transfers?" Both sides carry per-fixture captaincy + bench.
        hw = f"_hw{k}"
        fwd = [f"F{k + j} XP" for j in range(len(HORIZON_W)) if k + j <= config.HORIZON and f"F{k + j} XP" in m.columns]
        wts = HORIZON_W[:len(fwd)]
        _, ev_hxi = best_xi(ev, hw)                             # evolved squad's XI over the forward window
        wc_k = best_squad(m, budget, hw, current_idx, 15, want="both",      # unlimited => global-optimal window team
                          bench_slots=WILDCARD_BENCH_SLOTS, gk_bench=WILDCARD_GK_BENCH)
        if wc_k:
            w15, wxi = wc_k
            wc = round(max(0.0, _hz_value(m, w15, wxi, hw, fwd, wts, WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)
                           - _hz_value(m, ev15, ev_hxi, hw, fwd, wts, WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)), 2)
        else:                                                  # unlimited transfers can't lower value; a small
            wc = None                                          # negative is the captaincy-scoring approximation
        rows.append({"f": k, "ft": ft_avail[k], "bench_boost": round(bb, 2),
                     "triple_captain": round(tc, 2), "free_hit": fh, "wildcard": wc})
    return rows


def _print_radar(rows, gw):
    def mark(rows, key):                       # '*' on the standout week for this chip
        vals = [r[key] for r in rows if r[key] is not None]
        return max(vals) if vals else None
    best = {k: mark(rows, k) for k in ("bench_boost", "triple_captain", "free_hit", "wildcard")}
    print(f"\nchip radar — each of the next {len(rows)} gameweeks as a full-weight one-off (`*` = standout week):")
    print(f"  {'gameweek':<11}{'bench boost':>13}{'triple capt':>13}{'free hit':>11}{'wildcard':>11}{'(sim FTs)':>11}")
    for r in rows:
        cells = ""
        for key, w in (("bench_boost", 13), ("triple_captain", 13), ("free_hit", 11), ("wildcard", 11)):
            v = r[key]
            s = "-" if v is None else f"{v:.2f}"
            s += "*" if v is not None and v == best[key] else " "
            cells += f"{s:>{w}}"
        label = f"GW{gw + r['f'] - 1} (F{r['f']})"      # F{k} maps to calendar GW(gw+k-1)
        print(f"  {label:<11}{cells}{r['ft']:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, help="gameweek (defaults to inference; pass it to match the archive)")
    ap.add_argument("--season", default=config.SEASON)
    ap.add_argument("--free-transfers", type=int, default=1,
                    help="free transfers available - the baseline both deltas are measured against (default 1)")
    ap.add_argument("--bank", type=float, default=0.0, help="money in the bank, added to the rebuild budget")
    ap.add_argument("--budget", type=float, help="override the rebuild budget (default: squad value + bank)")
    ap.add_argument("--radar", action="store_true",
                    help="also print the forward F1..F8 chip radar (each fixture as a full-weight one-off)")
    ap.add_argument("--min-f1-gain", type=float, default=0.5,
                    help="baseline keeps its free transfer only if the F1 XI (incl captain) improves by "
                         ">= this, else holds (default 0.5; 0 = always transfer, old behaviour)")
    ap.add_argument("--min-total-gain", type=float, default=1.0,
                    help="baseline keeps its free transfer only if the weighted 8-fixture total (incl "
                         "bench) improves by >= this (default 1.0; 0 = always transfer)")
    ap.add_argument("--min-remaining-gain", type=float, default=1.0,
                    help="baseline keeps its free transfer only if the F2-onward (tail) weighted XI+"
                         "captain improves by >= this (default 1.0; blocks rent-shaped moves; 0 = off)")
    a = ap.parse_args()
    gw = a.gw or history.infer_gameweek()
    if not gw:
        raise SystemExit("could not infer gameweek — pass --gw N")
    ft = a.free_transfers

    radar_cols = [f"F{k} XP" for k in range(1, config.HORIZON + 1) if f"F{k} XP" in pd.read_csv(MASTER, nrows=0).columns]
    m = pd.read_csv(MASTER)[["Player Name", "Position", "Team", "Cost"] + radar_cols].copy()
    m["_htotal"] = sum(w * m[c] for w, c in zip(HORIZON_W, FIX_COLS))   # optimiser's 8-fixture total (F1-F8)
    names = current_squad_names(gw)          # the squad you carried INTO this GW (GW-1)
    squad = m[m["Player Name"].isin(names)]
    current_idx = list(squad.index)
    if len(squad) < 15:
        print(f"warning: only {len(squad)}/15 squad players matched the master "
              f"(missing {sorted(set(names) - set(squad['Player Name']))})")

    # Baseline = the NORMAL team you'd field this GW: best team reachable with `ft` free transfers from
    # the squad you carried in, on the 8-fixture (ownership x reliability) objective. ALL chips are
    # measured against this SAME baseline: BENCH BOOST / TRIPLE CAPTAIN are single-GW (F1) on the
    # baseline's own 15; WILDCARD is a 15-transfer rebuild over 8 fixtures; FREE HIT a 15-transfer
    # rebuild on F1. Captaincy is on both sides.
    budget = a.budget if a.budget is not None else float(squad["Cost"].sum()) + a.bank
    _gbase = gated_best_squad(m, budget, current_idx, ft, a.min_f1_gain, a.min_total_gain,  # gated normal team
                              a.min_remaining_gain, WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)
    base = _gbase[:2] if _gbase else None
    base_kept = _gbase[2] if _gbase else True
    wc = best_squad(m, budget, "_htotal", current_idx, 15, want="both",         # wildcard: horizon rebuild
                    bench_slots=WILDCARD_BENCH_SLOTS, gk_bench=WILDCARD_GK_BENCH)
    fh = best_squad(m, budget, "F1 XP", current_idx, 15, want="both",           # free hit: F1 rebuild (thin bench)
                    bench_slots=FREE_HIT_BENCH_SLOTS, gk_bench=FREE_HIT_GK_BENCH)

    if None in (base, wc, fh):
        # No PuLP: value the single-GW chips on the carried-in squad (best-effort), skip the ILP deltas.
        _, f1_xi = best_xi(squad, "F1 XP")
        bench_boost = float(squad.loc[~squad.index.isin(f1_xi), "F1 XP"].sum())
        triple_captain = float(squad.loc[list(f1_xi), "F1 XP"].max())
        print("(Wildcard/Free Hit deltas skipped — PuLP unavailable; run with env/Scripts/python)")
        wildcard_diff = free_hit_diff = ""
        base_h_tot = wc_tot = base_f_tot = fh_tot = None
    else:
        base_15, base_h = base
        wc_15, wc_xi = wc
        fh_15, fh_xi = fh
        base_sq = m.loc[list(base_15)]                                    # the ft-adjusted normal 15
        _, f1_xi = best_xi(base_sq, "F1 XP")                              # the XI you'd actually field (F1)
        bench_boost = float(base_sq.loc[~base_sq.index.isin(f1_xi), "F1 XP"].sum())
        triple_captain = float(base_sq.loc[list(f1_xi), "F1 XP"].max())   # the +1x on your captain
        # WILDCARD: XI + captain + bench (permanent-squad slots), both sides on the 8-fixture objective.
        base_h_tot = horizon_total(m, base_h) + bench_value(m, base_15, base_h, "_htotal", WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)
        wc_tot = horizon_total(m, wc_xi) + bench_value(m, wc_15, wc_xi, "_htotal", WILDCARD_BENCH_SLOTS, WILDCARD_GK_BENCH)
        # FREE HIT: F1 XI + captain + bench (one-week slots), both sides on F1.
        base_f_tot = f1_total(m, base_h) + bench_value(m, base_15, base_h, "F1 XP", FREE_HIT_BENCH_SLOTS, FREE_HIT_GK_BENCH)
        fh_tot = f1_total(m, fh_xi) + bench_value(m, fh_15, fh_xi, "F1 XP", FREE_HIT_BENCH_SLOTS, FREE_HIT_GK_BENCH)
        wildcard_diff = round(wc_tot - base_h_tot, 2)
        free_hit_diff = round(fh_tot - base_f_tot, 2)

    highest_xp = round(float(m["F1 XP"].max()), 2)     # highest projected F1 XP of ANY player this GW (league max)

    row = {"Season": a.season, "Gameweek": gw,
           "free_transfers": ft,
           "bench_boost": round(bench_boost, 2),
           "triple_captain": round(triple_captain, 2),
           "highest_xp": highest_xp,
           "wildcard_diff": wildcard_diff,
           "free_hit_diff": free_hit_diff,
           "baseline_f1": round(base_f_tot, 2) if base_f_tot is not None else "",
           "free_hit_f1": round(fh_tot, 2) if fh_tot is not None else "",
           "budget": round(budget, 1)}

    if os.path.exists(HIST):
        h = pd.read_csv(HIST)
        h = h[~((h["Season"].astype(str) == str(a.season)) & (h["Gameweek"] == gw))]
        h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
    else:
        h = pd.DataFrame([row])
    h = h.sort_values(["Season", "Gameweek"]).reset_index(drop=True)
    h.to_csv(HIST, index=False)

    if a.min_f1_gain > 0 or a.min_total_gain > 0 or a.min_remaining_gain > 0:
        print(f"\nBaseline gate: keep the free transfer only if F1 XI improves >= +{a.min_f1_gain} AND "
              f"F2-onward (tail) weighted XI improves >= +{a.min_remaining_gain} AND weighted total "
              f"(incl bench) improves >= +{a.min_total_gain} — "
              f"{'baseline MADE its transfer' if base_kept else 'baseline HELD (0 transfers)'}")
    print(f"\nGW{gw} {a.season} (baseline = {ft} free transfer{'s' if ft != 1 else ''}):"
          f"\n  bench boost    = {bench_boost:.2f} (F1)"
          f"\n  triple captain = {triple_captain:.2f} (F1)"
          f"\n  highest XP     = {highest_xp:.2f} (any player, F1)"
          f"\n  wildcard delta = {wildcard_diff} (8-fixture horizon"
          + (f": baseline {base_h_tot:.1f} -> rebuild {wc_tot:.1f})" if wc_tot is not None else ")")
          + f"\n  free hit delta = {free_hit_diff} (F1 only"
          + (f": baseline {base_f_tot:.1f} -> rebuild {fh_tot:.1f})" if fh_tot is not None else ")"))
    print(f"\nlogged to {os.path.relpath(HIST, ROOT)} — all logged weeks:")
    print(h.to_string(index=False, na_rep=""))

    if a.radar:
        _print_radar(chip_radar(m, squad, current_idx, budget, ft, a.min_f1_gain, a.min_total_gain,
                                a.min_remaining_gain), gw)


if __name__ == "__main__":
    main()
