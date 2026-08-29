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
import datetime
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, history  # noqa: E402

MASTER = os.path.join(ROOT, "outputs", "13_players_master.csv")
GW_TEAMS = os.path.join(ROOT, "inputs", "gw_teams.csv")
HIST = os.path.join(ROOT, "inputs", "chip_history.csv")
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FIX_COLS = [f"F{k} XP" for k in range(1, 9)]
# Optimiser's 8-fixture weights (ownership × reliability, F1-normalised) — replicated from
# optimisation.py rather than imported, because importing that module fires its top-level run.
_OWN = (1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558)
_REL = (1.0, 0.850, 0.616, 0.578, 0.503, 0.498, 0.481, 0.472)
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


def best_squad(pool, budget, col, current_idx=None, max_transfers=15, want="xi"):
    """Indices of the best legal XI maximising the sum of `col` over an 11-man XI inside a 15-man
    squad, within `budget` and <=3 players per team, changing AT MOST `max_transfers` players from
    `current_idx` (None / 15 = a free rebuild). `want`: "xi" -> the 11 starters (default),
    "squad" -> the 15-man squad, "both" -> (squad_set, xi_set). Returns None if PuLP is missing /
    the problem is infeasible. Captaincy is added post-hoc by the *_total helpers."""
    try:
        import pulp
    except ImportError:
        return None
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    idx = list(pool.index)
    sq = {i: pulp.LpVariable(f"sq_{i}", cat="Binary") for i in idx}
    st = {i: pulp.LpVariable(f"st_{i}", cat="Binary") for i in idx}
    prob += pulp.lpSum(pool.loc[i, col] * st[i] for i in idx)
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


MAX_FT = 5   # FPL caps banked free transfers at 5


def chip_radar(m, squad, current_idx, budget, ft):
    """Forward radar: treat EACH of F1..F8 as a standalone, full-weight one-off and value a Bench
    Boost / Triple Captain / Free Hit there — so a spike anywhere in the 8-fixture window is visible
    in advance, not only when it becomes F1. Bench Boost/Triple Captain read the current squad;
    Free Hit is the per-fixture 15-transfer rebuild (optimised for THAT fixture) minus the NORMAL
    team you'd otherwise field — the horizon-optimised (`_htotal`) team reachable with your
    ACCUMULATED free transfers by then (`ft` at F1, ft+1 at F2, ... capped at MAX_FT) — evaluated on
    that fixture. So the baseline is the realistic team you'd have, not one myopically built for the
    fixture. Note: Ff slots are per-FIXTURE (not calendar), so double/blank GWs map only approximately."""
    # Bench Boost / Triple Captain read the team you hold NOW — the carried-in squad plus your `ft` free
    # transfers (horizon-optimised) — held fixed and valued on each fixture. So F1 matches the headline,
    # and later weeks show which upcoming fixture YOUR bench spikes on (not a fodder-bench rebuild).
    now15 = best_squad(m, budget, "_htotal", current_idx, ft, want="squad")
    now = m.loc[list(now15)] if now15 else squad
    rows, base_by_ft = [], {}
    for k in range(1, 9):
        col = f"F{k} XP"
        _, xi = best_xi(now, col)
        bb = float(now.loc[~now.index.isin(xi), col].sum())
        tc = float(now.loc[list(xi), col].max())
        bt = min(ft + (k - 1), MAX_FT)                          # accumulated free transfers by F{k}
        if bt not in base_by_ft:                                # normal team for this transfer count (horizon obj)
            base_by_ft[bt] = best_squad(m, budget, "_htotal", current_idx, bt)
        base_k, fh_k = base_by_ft[bt], best_squad(m, budget, col, current_idx, 15)
        fh = round(fk_total(m, fh_k, col) - fk_total(m, base_k, col), 2) if base_k and fh_k else None
        rows.append({"f": k, "ft": bt, "bench_boost": round(bb, 2),
                     "triple_captain": round(tc, 2), "free_hit": fh})
    return rows


def _print_radar(rows, gw):
    def mark(rows, key):                       # '*' on the standout week for this chip
        vals = [r[key] for r in rows if r[key] is not None]
        return max(vals) if vals else None
    best = {k: mark(rows, k) for k in ("bench_boost", "triple_captain", "free_hit")}
    print("\nchip radar — each of the next 8 gameweeks as a full-weight one-off (`*` = standout week):")
    print(f"  {'gameweek':<11}{'bench boost':>13}{'triple capt':>13}{'free hit':>11}{'(FT base)':>11}")
    for r in rows:
        cells = ""
        for key, w in (("bench_boost", 13), ("triple_captain", 13), ("free_hit", 11)):
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
    a = ap.parse_args()
    gw = a.gw or history.infer_gameweek()
    if not gw:
        raise SystemExit("could not infer gameweek — pass --gw N")
    ft = a.free_transfers

    m = pd.read_csv(MASTER)[["Player Name", "Position", "Team", "Cost"] + FIX_COLS].copy()
    m["_htotal"] = sum(w * m[c] for w, c in zip(HORIZON_W, FIX_COLS))   # optimiser's 8-fixture total
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
    base = best_squad(m, budget, "_htotal", current_idx, ft, want="both")  # (15-man squad, XI) normal team
    wc_xi = best_squad(m, budget, "_htotal", current_idx, 15)              # wildcard: full rebuild, horizon obj
    fh_xi = best_squad(m, budget, "F1 XP", current_idx, 15)                # free hit: full rebuild, F1 obj

    if None in (base, wc_xi, fh_xi):
        # No PuLP: value the single-GW chips on the carried-in squad (best-effort), skip the ILP deltas.
        _, f1_xi = best_xi(squad, "F1 XP")
        bench_boost = float(squad.loc[~squad.index.isin(f1_xi), "F1 XP"].sum())
        triple_captain = float(squad.loc[list(f1_xi), "F1 XP"].max())
        print("(Wildcard/Free Hit deltas skipped — PuLP unavailable; run with env/Scripts/python)")
        wildcard_diff = free_hit_diff = ""
        base_h_tot = wc_tot = base_f_tot = fh_tot = None
    else:
        base_15, base_h = base
        base_sq = m.loc[list(base_15)]                                    # the ft-adjusted normal 15
        _, f1_xi = best_xi(base_sq, "F1 XP")                              # the XI you'd actually field (F1)
        bench_boost = float(base_sq.loc[~base_sq.index.isin(f1_xi), "F1 XP"].sum())
        triple_captain = float(base_sq.loc[list(f1_xi), "F1 XP"].max())   # the +1x on your captain
        base_h_tot, wc_tot = horizon_total(m, base_h), horizon_total(m, wc_xi)
        base_f_tot, fh_tot = f1_total(m, base_h), f1_total(m, fh_xi)      # same baseline team, valued on F1
        wildcard_diff = round(wc_tot - base_h_tot, 2)
        free_hit_diff = round(fh_tot - base_f_tot, 2)

    row = {"Season": a.season, "Gameweek": gw,
           "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "free_transfers": ft,
           "bench_boost": round(bench_boost, 2),
           "triple_captain": round(triple_captain, 2),
           "wildcard_diff": wildcard_diff,
           "free_hit_diff": free_hit_diff,
           "baseline_horizon": round(base_h_tot, 2) if base_h_tot is not None else "",
           "wildcard_horizon": round(wc_tot, 2) if wc_tot is not None else "",
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

    print(f"\nGW{gw} {a.season} (baseline = {ft} free transfer{'s' if ft != 1 else ''}):"
          f"\n  bench boost    = {bench_boost:.2f} (F1)"
          f"\n  triple captain = {triple_captain:.2f} (F1)"
          f"\n  wildcard delta = {wildcard_diff} (8-fixture horizon"
          + (f": baseline {base_h_tot:.1f} -> rebuild {wc_tot:.1f})" if wc_tot is not None else ")")
          + f"\n  free hit delta = {free_hit_diff} (F1 only"
          + (f": baseline {base_f_tot:.1f} -> rebuild {fh_tot:.1f})" if fh_tot is not None else ")"))
    print(f"\nlogged to {os.path.relpath(HIST, ROOT)} — recent weeks:")
    print(h.tail(10).to_string(index=False))

    if a.radar:
        _print_radar(chip_radar(m, squad, current_idx, budget, ft), gw)


if __name__ == "__main__":
    main()
