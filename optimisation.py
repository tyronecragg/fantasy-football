import os
import sys
import warnings

import pandas as pd
import pulp

# The two-stage tie-break deliberately re-sets the objective (XP ceiling, then ownership);
# PuLP warns on every objective overwrite, which is expected here, not a problem.
warnings.filterwarnings("ignore", message="Overwriting previously set objective")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_pipeline.prices import apply_sell_prices  # noqa: E402


def _purchase_csv(source):
    source = str(source)
    root = (os.path.dirname(os.path.dirname(os.path.abspath(source)))
            if source.lower().endswith('.csv') else os.path.dirname(os.path.abspath(source)) or '.')
    return os.path.join(root, 'inputs', 'purchase_prices.csv')


def load_sheet(source, sheet_name='Players'):
    """Load a data table from the legacy Excel workbook or the fpl_pipeline CSVs.

    Pass the workbook path (.xlsx) to read the named sheet, or the pipeline's master
    players CSV (outputs/13_players_master.csv). With a CSV source, the 'GW Teams'
    sheet is read from inputs/gw_teams.csv in the repo root.
    """
    source = str(source)
    if source.lower().endswith('.csv'):
        if sheet_name == 'GW Teams':
            root = os.path.dirname(os.path.dirname(os.path.abspath(source)))
            return pd.read_csv(os.path.join(root, 'inputs', 'gw_teams.csv'))
        return pd.read_csv(source)
    return pd.read_excel(source, sheet_name=sheet_name)

DGW_TEAMS = {}
DGW_EXTRA = True
DGW_EXTRA_FACTOR = 1


def apply_dgw_adjustment(df, dgw_teams=DGW_TEAMS, dgw_extra=DGW_EXTRA, dgw_extra_factor=DGW_EXTRA_FACTOR):
    """
    For players whose team is in dgw_teams:
      - F1 XP := F1 XP + F2 XP
      - F2 XP := 0
    Modifies df in place and also returns it for convenience.
    Safe to call even if 'F2 XP' isn't present.
    """
    if 'F1 XP' not in df.columns or 'F2 XP' not in df.columns:
        return df
    if 'Team' not in df.columns:
        return df

    mask = df['Team'].isin(dgw_teams)
    if mask.any():
        if dgw_extra:
            df.loc[mask, 'F1 XP'] = df.loc[mask, 'F1 XP']*(1+dgw_extra_factor)
        else:
            df.loc[mask, 'F1 XP'] = df.loc[mask, 'F1 XP'] + df.loc[mask, 'F2 XP']
            df.loc[mask, 'F2 XP'] = 0

        affected = df.loc[mask, 'Player Name'].tolist() if 'Player Name' in df.columns else []
        print(f"[DGW adjustment] Applied to {len(affected)} players from {sorted(dgw_teams)}")
    return df
# ============================================================
# END TEMPORARY BLOCK
# ============================================================


def _unavailable_csv(source):
    source = str(source)
    root = (os.path.dirname(os.path.dirname(os.path.abspath(source)))
            if source.lower().endswith('.csv') else os.path.dirname(os.path.abspath(source)) or '.')
    return os.path.join(root, 'inputs', 'unavailable_players.csv')


def departed_players(source):
    """Names of players who have permanently LEFT the league, read from
    inputs/unavailable_players.csv (reason contains 'left' or 'permanent'). Unlike injuries
    or in-progress 'being sold' notes — which stay selectable — these cannot be picked at
    all. 'GW1 hold' sale risks (a player deliberately kept for now) are NOT matched.
    """
    path = _unavailable_csv(source)
    if not os.path.exists(path):
        return set()
    u = pd.read_csv(path)
    if 'reason' not in u.columns or 'Player' not in u.columns:
        return set()
    gone = u['reason'].astype(str).str.contains(r'left|permanent', case=False, na=False)
    return {n.strip() for n in u.loc[gone, 'Player'].astype(str)}


def drop_departed(df, current_team_names, departed):
    """Remove permanently-departed players from the candidate pool — except any the manager
    still owns, which must remain so they can be transferred out."""
    owned = {n.strip() for n in current_team_names}
    names = df['Player Name'].astype(str).str.strip()
    gone = names.isin(departed) & ~names.isin(owned)
    if gone.any():
        print(f"  Excluded {int(gone.sum())} departed player(s): "
              + ", ".join(sorted(df.loc[gone, 'Player Name'])))
    return df[~gone]


GENERIC_TEAM = "(any)"   # sentinel team for collapsed fillers; exempt from per-team caps


def collapse_fungible_bench(df, current_team_names, fixture_columns):
    """A non-playing bench filler is a fungible commodity: among players at one position who
    are projected to score ~0 across the WHOLE horizon, only price matters, so the optimiser
    is indifferent between them and the solution enumerator burns near-optimal solutions
    swapping one warm body for another (observed: 25 'distinct' squads differing only in a
    £4.0m 0-XP keeper).

    Per position, collapse the 0-XP crowd to a SINGLE cheapest representative, relabelled
    generically ('Any £4.0m 0-XP DEF'). Owned players are never dropped (they must stay
    transferable). But if the manager ALREADY OWNS a 0-XP filler here, that filler IS the
    slot: a bought generic is only worth offering when it is STRICTLY cheaper than the owned
    one (a real budget-freeing downgrade). A same/higher-price generic is a wasted transfer
    for nothing, so it is dropped and the owned incumbent stands un-churned (otherwise the
    enumerator flips between 'keep your £4.0m keeper' and 'buy an identical £4.0m keeper').

    The optimum is unchanged: the cheapest 0-XP filler was always what it wanted, and for
    outfield slots the optimiser prefers a cheap PLAYING player anyway (bench points), so real
    cheap options remain to keep every squad feasible. A player with 0 XP now but real XP later
    (injury ramp) has max fixture XP > 0 and is never collapsed. Returns (df, {pos: name}).
    """
    owned = {n.strip() for n in current_team_names}
    never_plays = df[fixture_columns].max(axis=1) < 0.05
    owned_mask = df['Player Name'].astype(str).str.strip().isin(owned)
    pos_norm = df['Position'].astype(str).str.upper().replace({'GKP': 'GK'})
    generics = {}
    df = df.copy()
    drop = []
    for pos in ('GK', 'DEF', 'MID', 'FWD'):
        fungible = (pos_norm == pos) & never_plays
        owned_fung = df.index[fungible & owned_mask]
        non_owned = list(df.index[fungible & ~owned_mask])
        if len(owned_fung):
            # Owned filler is the incumbent: keep only non-owned fillers strictly cheaper
            # than it (a genuine downgrade); drop the same/higher-price ones as pure churn.
            floor = df.loc[owned_fung, 'Cost'].min()
            drop.extend(i for i in non_owned if df.loc[i, 'Cost'] >= floor)
            non_owned = [i for i in non_owned if df.loc[i, 'Cost'] < floor]
            min_to_collapse = 1     # even one surviving downgrade becomes the generic option
        else:
            min_to_collapse = 2     # no incumbent: collapse only when it would otherwise churn
        if len(non_owned) >= min_to_collapse:
            rep = min(non_owned, key=lambda i: df.loc[i, 'Cost'])
            name = f"Any £{df.loc[rep, 'Cost']:.1f}m 0-XP {pos}"
            df.loc[rep, 'Player Name'] = name
            df.loc[rep, 'Team'] = GENERIC_TEAM   # no committed team - exempt from per-team caps
            generics[pos] = name
            drop.extend(i for i in non_owned if i != rep)
    if drop:
        df = df.drop(index=drop)
        print("  Collapsed interchangeable 0-XP bench fillers"
              + (f" ({', '.join(generics.values())})" if generics else " (owned incumbents kept)")
              + f"; dropped {len(drop)} redundant duplicates")
    return df, generics


def load_current_team(excel_file, sheet_name='GW Teams'):
    print("Loading current team from GW Teams sheet...")

    # Read the GW Teams sheet
    df_teams = load_sheet(excel_file, sheet_name)

    # Find the rightmost column with data (current gameweek)
    last_col_idx = df_teams.shape[1] - 1
    while last_col_idx > 0 and df_teams.iloc[:, last_col_idx].isna().all():
        last_col_idx -= 1

    current_gw_column = df_teams.columns[last_col_idx]
    current_team_names = df_teams.iloc[:, last_col_idx].dropna().tolist()

    print(f"Found current team for {current_gw_column}: {len(current_team_names)} players")
    for i, player in enumerate(current_team_names, 1):
        print(f"  {i:2d}. {player}")

    return current_team_names, current_gw_column


def calculate_current_team_value(excel_file, current_team_names, players_sheet='Players'):
    """
    Calculate the current market value of the team

    Parameters:
    - excel_file: Path to Excel file
    - current_team_names: List of current player names
    - players_sheet: Sheet name with player data

    Returns:
    - total_value: Sum of current player values
    - player_values: Dict of player names to their values
    """
    # Load player data
    df = load_sheet(excel_file, players_sheet)
    df.columns = df.columns.str.strip()
    apply_dgw_adjustment(df)  # TEMPORARY
    apply_sell_prices(df, current_team_names, _purchase_csv(excel_file))

    total_value = 0
    player_values = {}
    missing_players = []

    for player_name in current_team_names:
        # Try exact match first
        matches = df[df['Player Name'].str.strip() == player_name.strip()]

        if len(matches) == 0:
            # Try partial match if exact match fails
            matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]

        if len(matches) > 0:
            player_value = matches.iloc[0]['Cost']
            player_values[player_name] = player_value
            total_value += player_value
        else:
            missing_players.append(player_name)
            print(f"Warning: Could not find player '{player_name}' in database for value calculation")

    if missing_players:
        print(f"Missing players for value calculation: {missing_players}")

    print(f"Current team value: £{total_value:.1f}m")
    return total_value, player_values


def analyse_current_team(excel_file, current_team_names, num_fixtures=6, fixture_weights=None,
                         players_sheet='Players', additional_budget=0.0):
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    weights = fixture_weights[:num_fixtures]

    # Load player data
    df = load_sheet(excel_file, players_sheet)
    df.columns = df.columns.str.strip()
    apply_dgw_adjustment(df)  # TEMPORARY
    apply_sell_prices(df, current_team_names, _purchase_csv(excel_file))

    # Define fixture columns
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP', 'F7 XP', 'F8 XP']
    fixture_columns = all_fixture_columns[:num_fixtures]

    # Calculate weighted total XP
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

    # Find current team players in the database
    current_team_players = []
    missing_players = []

    for player_name in current_team_names:
        # Try exact match first
        matches = df[df['Player Name'].str.strip() == player_name.strip()]

        if len(matches) == 0:
            # Try partial match if exact match fails
            matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]

        if len(matches) > 0:
            current_team_players.append(matches.iloc[0])
        else:
            missing_players.append(player_name)
            print(f"Warning: Could not find player '{player_name}' in database")

    if missing_players:
        print(f"\nMissing players: {missing_players}")
        print("Please check player names or update the database")

    # Create current team DataFrame
    current_team_df = pd.DataFrame(current_team_players)

    if len(current_team_df) == 0:
        print("Error: No current team players found in database!")
        return None, 0, {}

    # Calculate team statistics
    team_value = current_team_df['Cost'].sum()
    total_budget = team_value + additional_budget
    total_weighted_points = current_team_df['Weighted_Total_XP'].sum()

    # Position breakdown
    position_counts = current_team_df['Position'].value_counts()

    # Team breakdown
    team_counts = current_team_df['Team'].value_counts()

    analysis = {
        'team_value': team_value,
        'additional_budget': additional_budget,
        'total_budget': total_budget,
        'total_weighted_points': total_weighted_points,
        'position_counts': position_counts,
        'team_counts': team_counts,
        'avg_points_per_fixture': total_weighted_points / num_fixtures,
        'num_players_found': len(current_team_df),
        'missing_players': missing_players
    }

    return current_team_df, total_weighted_points, analysis


def display_current_team_analysis(current_team_df, analysis, num_fixtures, weights):
    if current_team_df is None or len(current_team_df) == 0:
        print("No current team data to display!")
        return

    # Points summary
    print(f"\nPOINTS SUMMARY ({num_fixtures} fixtures):")
    print(f"  Total Weighted Points: {analysis['total_weighted_points']:.2f}")
    print(f"  Average per Fixture: {analysis['avg_points_per_fixture']:.2f}")
    print(f"  Fixture Weights: {[f'{w:.2f}' for w in weights]}")

    # Position requirements check
    print(f"\nPOSITION REQUIREMENTS:")
    required = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, req_count in required.items():
        actual_count = analysis['position_counts'].get(pos, 0)
        status = "✓" if actual_count == req_count else "✗"
        print(f"  {pos}: {actual_count}/{req_count} {status}")

    # Team limits check
    print(f"\nTEAM LIMITS (max 3 per team):")
    for team, count in analysis['team_counts'].items():
        status = "✓" if count <= 3 else "✗"
        print(f"  {team}: {count}/3 {status}")

    if analysis['missing_players']:
        print(f"\nMISSING PLAYERS:")
        for player in analysis['missing_players']:
            print(f"  - {player}")


# Outfield bench priced by sub order: slot 1 is the autosub workhorse (~25-30% of
# weeks), slot 2 covers double absences (~5%), slot 3 almost never comes on.
BENCH_SLOT_WEIGHTS = (0.30, 0.10, 0.05)

# --- Fixture weights are TWO separate discounts, kept apart because they have
# different causes, different evidence, and change on different schedules. ---
#
# 1. OWNERSHIP ("can I still fix it?"). A bad F6 fixture is recoverable with a
#    transfer; a bad F1 is not. So far fixtures are worth less regardless of how well
#    we forecast them. Decays at the squad churn rate: ~1.2 transfers/week over 15
#    players = 8%/gameweek, i.e. 0.92 ** (k - 1). Roughly constant across a season.
OWNERSHIP_WEIGHTS = (1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558)

# 2. RELIABILITY ("how much do we trust the projection?"). MEASURED from the backtest
#    archive as skill vs a positional-mean baseline (tools/backtest_projections.py,
#    horizon k -> F(k+1)): F2 0.468, F3 0.339, F4 0.318, F5 0.277, F6 0.274, then flat
#    because everything from F3 on shares one static team model. Normalised to F2 and
#    scaled by F2_VS_F1 below. The steep F2 -> F3 step is real: that is where the input
#    changes from market odds to model projection, so the curve is a CLIFF, not a ramp.
#    CAVEATS: the archive starts ~GW12, so these describe the mid-season regime only -
#    August outright markets are more diffuse and are not represented yet. F7/F8 are
#    extrapolated from the flat tail. F2's level rests on F2_VS_F1, an assumption, not a
#    measurement: if F2 usually has real odds scraped, raise it toward 1.0.
F2_VS_F1 = 0.85
RELIABILITY_WEIGHTS = (1.0, 0.850, 0.616, 0.578, 0.503, 0.498, 0.481, 0.472)


def combine_fixture_weights(ownership=None, reliability=None, num_fixtures=8):
    """Multiply the two discounts into the single weight vector the optimiser uses,
    normalised so F1 = 1.0. Pass either component to explore it in isolation."""
    own = list(ownership or OWNERSHIP_WEIGHTS)[:num_fixtures]
    rel = list(reliability or RELIABILITY_WEIGHTS)[:num_fixtures]
    if len(own) < num_fixtures or len(rel) < num_fixtures:
        raise ValueError(f"need {num_fixtures} ownership and reliability weights "
                         f"(got {len(own)} and {len(rel)})")
    combined = [o * r for o, r in zip(own, rel)]
    return [w / combined[0] for w in combined]


_TC_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_TC_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
_TC_SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _tc_best_squad(d, budget, obj, current_idx, max_transfers):
    """Best 15-squad maximising `obj` over its 11-man XI, within budget & <=3/club, changing at most
    `max_transfers` players from current_idx. Returns (xi_set, squad15_set) or (None, None). XI-only
    objective (bench weight 0) — a chip/timing lens, matching tools/chip_history."""
    idx = list(d.index)
    prob = pulp.LpProblem("tc", pulp.LpMaximize)
    sq = {i: pulp.LpVariable(f"tq{i}", cat="Binary") for i in idx}
    st = {i: pulp.LpVariable(f"ts{i}", cat="Binary") for i in idx}
    prob += pulp.lpSum(d.loc[i, obj] * st[i] for i in idx)
    prob += pulp.lpSum(sq.values()) == 15
    prob += pulp.lpSum(st.values()) == 11
    prob += pulp.lpSum(d.loc[i, "Cost"] * sq[i] for i in idx) <= budget
    for i in idx:
        prob += st[i] <= sq[i]
    for pos, n in _TC_SQUAD.items():
        p = d.index[d["Position"] == pos]
        prob += pulp.lpSum(sq[i] for i in p) == n
    for pos in _TC_XI_MIN:
        p = d.index[d["Position"] == pos]
        prob += pulp.lpSum(st[i] for i in p) >= _TC_XI_MIN[pos]
        prob += pulp.lpSum(st[i] for i in p) <= _TC_XI_MAX[pos]
    for team in d["Team"].dropna().unique():
        t = d.index[d["Team"] == team]
        prob += pulp.lpSum(sq[i] for i in t) <= 3
    if current_idx is not None and max_transfers < 15:
        prob += pulp.lpSum(sq[i] for i in current_idx) >= max(0, 15 - max_transfers)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None
    return ({i for i in idx if st[i].value() == 1}, {i for i in idx if sq[i].value() == 1})


def transfer_timing_check(excel_file, current_team_names, lookahead_gws, additional_budget=0.0,
                          free_transfers=1, max_per_gw=2, max_ft=5):
    """Transfer PLAN valued over the FULL 8-fixture horizon, where `lookahead_gws` only bounds WHEN
    transfers may be made (the timing window), NOT the evaluation window. It decides how many transfers
    to make each GW in F1..F{lookahead_gws} (0..max_per_gw, free-transfer accrual +1/GW capped at
    max_ft); the squad then evolves and is scored across ALL of F1..F8 (ownership x reliability weights).

    This is the key point: a myopic swap (sell a premium to dodge one bad fixture) LOSES that player's
    value across the rest of the horizon, so the full-horizon value rejects it — it only reorders WHEN
    you transfer, not who you'd never sell. Answers "use my free transfer now or bank it?" by searching
    every feasible plan and reporting THIS GW's optimal move count. lookahead_gws < 2 is a NO-OP (off).
    XI-only value (bench weight 0). Rolling — re-run each week, robust to forced (injury) transfers."""
    from tools.chip_history import best_xi
    K = int(lookahead_gws)
    if K < 2:
        return                                                          # =1 (or less) -> off
    N = len(OWNERSHIP_WEIGHTS)                                           # FULL horizon (8 fixtures)
    K = min(K, N)                                                        # transfers allowed only in GWs 0..K-1
    w = combine_fixture_weights(num_fixtures=N)
    fcols = [f"F{f} XP" for f in range(1, N + 1)]
    df = pd.read_csv(excel_file)
    apply_sell_prices(df, current_team_names, _purchase_csv(excel_file))
    d = df[["Player Name", "Position", "Team", "Cost"] + fcols].copy()
    for c in fcols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    for g in range(K):                                                  # a GW-g move is kept for F{g+1}..F8
        d[f"_obj{g}"] = sum(w[i] * d[fcols[i]] for i in range(g, N))
    owned = {n.strip() for n in current_team_names}
    cur = frozenset(d.index[d["Player Name"].isin(owned)].tolist())
    budget = float(d.loc[list(cur), "Cost"].sum()) + additional_budget

    def gw_xp(squad, f):                          # raw predicted XP at fixture index f: best XI + captain
        _, xi = best_xi(d.loc[list(squad)], fcols[f])
        x = d.loc[list(xi)]
        return float(x[fcols[f]].sum()) + float(x[fcols[f]].max())

    def moves(squad, base):
        return ([d.loc[i, "Player Name"] for i in base if i not in squad],
                [d.loc[i, "Player Name"] for i in squad if i not in base])

    best_by_t0 = {}                               # this-GW transfer count -> best plan starting with it

    def search(squad, g, ft, wacc, plan, raw):
        if g == K:                                                      # no more transfers; squad is final
            wtotal = wacc + sum(w[f] * gw_xp(squad, f) for f in range(K, N))   # tail fixtures F{K+1}..F8
            t0 = plan[0][0]
            if t0 not in best_by_t0 or wtotal > best_by_t0[t0]["wtotal"]:
                best_by_t0[t0] = {"wtotal": wtotal, "plan": list(plan), "raw": list(raw)}
            return
        for t in range(0, min(ft, max_per_gw) + 1):                    # transfers this GW
            if t == 0:
                sq_t, mv = squad, ([], [])
            else:
                _, sq = _tc_best_squad(d, budget, f"_obj{g}", list(squad), t)   # optimise the FULL remaining horizon
                if sq is None:
                    continue
                sq_t, mv = frozenset(sq), moves(frozenset(sq), squad)
            r = gw_xp(sq_t, g)                                          # fixture index g, post-transfer squad
            search(sq_t, g + 1, min(ft - t + 1, max_ft), wacc + w[g] * r, plan + [(t, mv)], raw + [r])

    search(cur, 0, free_transfers, 0.0, [], [])
    if not best_by_t0:
        print("\nTransfer-timing check: infeasible (skipped).")
        return

    best_t0 = max(best_by_t0, key=lambda k: best_by_t0[k]["wtotal"])
    print(f"\n{'=' * 66}\nTransfer-timing plan — value over FULL F1..F{N}, transfers within F1..F{K} "
          f"(start {free_transfers} FT, <= {max_per_gw}/GW, cap {max_ft}):")
    for t0 in sorted(best_by_t0):
        rec = best_by_t0[t0]
        raw = rec["raw"]
        label = "BANK (hold now)" if t0 == 0 else f"MOVE ({t0} now)"
        star = "  <-- best" if t0 == best_t0 else ""
        print(f"  {label:<16} F1 XP {raw[0]:6.2f}   F2 XP {raw[1]:6.2f}   "
              f"8-fixture value {rec['wtotal']:7.2f}{star}")
        for g, (t, (out, inn)) in enumerate(rec["plan"]):
            if t:
                tag = "now" if g == 0 else f"GW+{g}"
                print(f"       {tag:<6} {', '.join(out)} -> {', '.join(inn)}")
    verdict = ("BANK your free transfer this GW" if best_t0 == 0
               else f"MAKE {best_t0} transfer{'s' if best_t0 > 1 else ''} this GW")
    print(f"  -> {verdict}\n{'=' * 66}")


def _normalise_slot_weights(spec, num_fixtures):
    """Accept one (s1, s2, s3) triple applied to every fixture, or a per-fixture list
    of triples (e.g. [(1, 1, 1)] + [(0.3, 0.1, 0.05)] * 7 for a GW1 Bench Boost).
    A short per-fixture list is padded by repeating its last triple."""
    spec = list(spec)
    if spec and isinstance(spec[0], (int, float)):
        return [tuple(spec)] * num_fixtures
    per_fixture = [tuple(t) for t in spec][:num_fixtures]
    if per_fixture and len(per_fixture) < num_fixtures:
        per_fixture += [per_fixture[-1]] * (num_fixtures - len(per_fixture))
    return per_fixture


def _bench_slot_objective(prob, xp, bench_f, outfield_indices, weight, slot_weights, tag):
    """Objective terms pricing the outfield bench by sub order (slot 1 > 2 > 3).

    Sum-of-top-k encoding: every benched outfielder earns the slot-3 weight, the best
    two additionally earn (slot2 - slot3), and the best one (slot1 - slot2). The
    continuous helper variables are driven onto the highest-XP bench players by the
    maximisation itself, so no explicit ordering constraints are needed.
    """
    w1, w2, w3 = slot_weights
    assert w1 >= w2 >= w3 >= 0, "bench slot weights must be decreasing and non-negative"
    terms = [xp[i] * weight * w3 * bench_f[i] for i in outfield_indices]
    top1 = {i: pulp.LpVariable(f"bslot1_{tag}_{i}", lowBound=0, upBound=1) for i in outfield_indices}
    top2 = {i: pulp.LpVariable(f"bslot2_{tag}_{i}", lowBound=0, upBound=1) for i in outfield_indices}
    for i in outfield_indices:
        prob += top1[i] <= bench_f[i]
        prob += top2[i] <= bench_f[i]
    prob += pulp.lpSum(top1.values()) <= 1
    prob += pulp.lpSum(top2.values()) <= 2
    terms += [xp[i] * weight * (w1 - w2) * top1[i] for i in outfield_indices]
    terms += [xp[i] * weight * (w2 - w3) * top2[i] for i in outfield_indices]
    return terms


def bench_points_for_fixture(df, bench_indices, fixture_col, bench_slot_weights, gk_bench_weight):
    """Weighted bench value for one fixture: outfielders priced by sub order
    (best XP = slot 1), the backup GK priced separately."""
    gks = [i for i in bench_indices if df.loc[i, 'Position'] == 'GK']
    outfield = sorted((i for i in bench_indices if df.loc[i, 'Position'] != 'GK'),
                      key=lambda i: df.loc[i, fixture_col], reverse=True)
    total = sum(df.loc[i, fixture_col] * gk_bench_weight for i in gks)
    total += sum(df.loc[i, fixture_col] * w for i, w in zip(outfield, bench_slot_weights))
    return total


def pure_xp_lineup(squad_indices, df, fixture_col):
    """Max-XP legal XI, bench, and captain for one fixture, chosen PURELY on expected points
    from a fixed 15-man squad. Applied after the squad is decided so the ownership tie-break
    only ever shapes transfers/squad selection -- never who you field. Formation: 1 GK,
    3-5 DEF, 2-5 MID, 1-3 FWD; bench outfielders and the captain fall out by XP too.
    """
    pos = {k: [] for k in ('GK', 'DEF', 'MID', 'FWD')}
    for i in squad_indices:
        pos[df.loc[i, 'Position']].append(i)
    for k in pos:
        pos[k].sort(key=lambda i: df.loc[i, fixture_col], reverse=True)
    xi = pos['GK'][:1] + pos['DEF'][:3] + pos['MID'][:2] + pos['FWD'][:1]
    rest = pos['DEF'][3:5] + pos['MID'][2:5] + pos['FWD'][1:3]
    rest.sort(key=lambda i: df.loc[i, fixture_col], reverse=True)
    xi += rest[:4]
    xi_set = set(xi)
    bench = [i for i in squad_indices if i not in xi_set]
    captain = max(xi, key=lambda i: df.loc[i, fixture_col])
    return xi, bench, captain


def calculate_optimised_baseline(df, current_team_indices, fixtures, weights, bench_slot_weights, gk_bench_weights):
    """
    Calculate the optimal baseline score for the current squad by running a mini-optimizer.
    This simulates dynamic lineup selection across fixtures without any transfers.

    Parameters:
    - bench_slot_weights: (slot1, slot2, slot3) outfield bench weights by sub order,
      or a per-fixture list of such triples
    - gk_bench_weights: array of GK bench weights for each fixture

    Returns: (total_weighted_points, total_starting_xi_points, f1_squad_total, f1_starting_xi_total)
    """
    bench_slot_weights = _normalise_slot_weights(bench_slot_weights, len(fixtures))
    # Create a mini optimization problem for just lineup selection
    prob = pulp.LpProblem("Baseline_Lineup_Optimization", pulp.LpMaximize)

    # Decision variables - only for current squad players
    # Weekly starting XI
    starting_vars = {}
    for fixture in fixtures:
        starting_vars[fixture] = {}
        for i in current_team_indices:
            starting_vars[fixture][i] = pulp.LpVariable(f"base_start_{fixture}_{i}", cat='Binary')

    # Weekly captains
    captain_vars = {}
    for fixture in fixtures:
        captain_vars[fixture] = {}
        for i in current_team_indices:
            captain_vars[fixture][i] = pulp.LpVariable(f"base_cap_{fixture}_{i}", cat='Binary')

    # Bench players
    bench_vars = {}
    for fixture in fixtures:
        bench_vars[fixture] = {}
        for i in current_team_indices:
            bench_vars[fixture][i] = pulp.LpVariable(f"base_bench_{fixture}_{i}", cat='Binary')

    # Objective: maximize weighted points
    objective_terms = []
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        gk_bench_weight = gk_bench_weights[i]

        for player_idx in current_team_indices:
            # Starting points
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            # Captain bonus
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # GK bench value (outfield bench is priced by sub order below)
            if df.loc[player_idx, 'Position'] == 'GK':
                objective_terms.append(
                    df.loc[player_idx, fixture_col] * weight * gk_bench_weight * bench_vars[fixture][player_idx])

        outfield = [p for p in current_team_indices if df.loc[p, 'Position'] != 'GK']
        objective_terms.extend(_bench_slot_objective(
            prob, df[fixture_col], bench_vars[fixture], outfield, weight,
            bench_slot_weights[i], f"base_{fixture}"))

    prob += pulp.lpSum(objective_terms)

    # Constraints
    starting_position_requirements = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}

    for fixture in fixtures:
        # Exactly 11 starters
        prob += pulp.lpSum([starting_vars[fixture][i] for i in current_team_indices]) == 11

        # Position requirements (minimum)
        for position, min_count in starting_position_requirements.items():
            position_players = [i for i in current_team_indices if df.loc[i, 'Position'] == position]
            prob += pulp.lpSum([starting_vars[fixture][i] for i in position_players]) >= min_count

        # Maximum position constraints
        gk_players = [i for i in current_team_indices if df.loc[i, 'Position'] == 'GK']
        def_players = [i for i in current_team_indices if df.loc[i, 'Position'] == 'DEF']
        mid_players = [i for i in current_team_indices if df.loc[i, 'Position'] == 'MID']
        fwd_players = [i for i in current_team_indices if df.loc[i, 'Position'] == 'FWD']

        prob += pulp.lpSum([starting_vars[fixture][i] for i in gk_players]) <= 1
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) <= 5
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) <= 5
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) <= 3

        # Exactly 1 captain
        prob += pulp.lpSum([captain_vars[fixture][i] for i in current_team_indices]) == 1

        # Captain must be a starter
        for i in current_team_indices:
            prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

        # Exactly 4 bench players
        prob += pulp.lpSum([bench_vars[fixture][i] for i in current_team_indices]) == 4

        # Bench logic
        for i in current_team_indices:
            prob += bench_vars[fixture][i] <= (1 - starting_vars[fixture][i])
            prob += bench_vars[fixture][i] >= 1 - starting_vars[fixture][i] - 0

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.LpStatusOptimal:
        print("Warning: Baseline optimizer failed, using simple calculation")
        return None

    # Extract results
    total_starting_points = 0
    total_captain_points = 0
    total_bench_points = 0

    f1_starting = 0
    f1_captain = 0
    f1_bench = 0

    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]

        for player_idx in current_team_indices:
            if starting_vars[fixture][player_idx].varValue == 1:
                total_starting_points += df.loc[player_idx, fixture_col] * weight
                if i == 0:  # F1
                    f1_starting += df.loc[player_idx, fixture_col]

            if captain_vars[fixture][player_idx].varValue == 1:
                total_captain_points += df.loc[player_idx, fixture_col] * weight
                if i == 0:  # F1
                    f1_captain += df.loc[player_idx, fixture_col]

        bench_idx = [p for p in current_team_indices if bench_vars[fixture][p].varValue == 1]
        fixture_bench_points = bench_points_for_fixture(
            df, bench_idx, fixture_col, bench_slot_weights[i], gk_bench_weights[i])
        total_bench_points += fixture_bench_points * weight
        if i == 0:  # F1
            f1_bench = fixture_bench_points

    total_squad_weighted = total_starting_points + total_captain_points + total_bench_points
    total_starting_xi_weighted = total_starting_points + total_captain_points

    f1_squad_total = f1_starting + f1_captain + f1_bench
    f1_starting_xi_total = f1_starting + f1_captain

    return total_squad_weighted, total_starting_xi_weighted, f1_squad_total, f1_starting_xi_total


def optimise_transfers_multi(excel_file, current_team_names, max_transfers=2, num_fixtures=5,
                             fixture_weights=None, players_sheet='Players',
                             additional_budget=0.0, bench_slot_weights=None, gk_bench_weights=None,
                             force_transfer_out=None, num_solutions=3, max_defensive_players_per_team=3,
                             force_transfer_in=None, tie_breaker=None,
                             tie_break_mode='differential', xp_tolerance=0.5, value_weight=0.0):
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    # Outfield bench weights by sub order (slot 1/2/3); GK bench stays per-fixture
    if bench_slot_weights is None:
        bench_slot_weights = BENCH_SLOT_WEIGHTS

    if gk_bench_weights is None:
        gk_bench_weights = [0.10, 0.10, 0.08, 0.06, 0.04, 0.02]

    weights = fixture_weights[:num_fixtures]
    gk_bench_weights = gk_bench_weights[:num_fixtures]
    bench_slot_weights = _normalise_slot_weights(bench_slot_weights, num_fixtures)

    # Handle forced transfers
    if force_transfer_out is None:
        force_transfer_out = []
    if force_transfer_in is None:
        force_transfer_in = []

    print(f"\nOptimising transfers: max {max_transfers} transfers for {num_fixtures} fixtures")
    print(f"Finding top {num_solutions} transfer combinations")
    print(f"Using fixture weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional budget available: £{additional_budget:.1f}m")
    if value_weight:
        print(f"Value preference: {value_weight:.2f} weighted XP per £1.0m of squad value "
              f"(accepts {value_weight * 0.1:.2f} lower XP per £0.1m saved)")
    if len(set(bench_slot_weights)) == 1:
        print(f"Bench slot weights (sub order 1/2/3): {[f'{w:.2f}' for w in bench_slot_weights[0]]}")
    else:
        print(f"Bench slot weights (sub order 1/2/3, per fixture): {bench_slot_weights}")
    print(f"GK bench weights: {[f'{w:.2f}' for w in gk_bench_weights]}")
    print(f"Max defensive players (GK+DEF) per team: {max_defensive_players_per_team}")
    if force_transfer_out:
        print(f"Forced transfers out: {force_transfer_out}")
    if force_transfer_in:
        print(f"Forced transfers in: {force_transfer_in}")

    # Load player data
    df = load_sheet(excel_file, players_sheet)
    df.columns = df.columns.str.strip()
    apply_dgw_adjustment(df)  # TEMPORARY
    apply_sell_prices(df, current_team_names, _purchase_csv(excel_file))

    # Define fixture columns and fixtures
    fixtures = [f'F{i + 1}' for i in range(num_fixtures)]
    fixture_columns = [f'{fixture} XP' for fixture in fixtures]

    # Calculate weighted total XP
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

    # Clean the pool before optimising: drop players who have permanently left, and collapse
    # the interchangeable non-playing backup keepers to one generic option so near-optimal
    # solutions differ in choices that actually matter rather than in the bench-GK warm body.
    df = drop_departed(df, current_team_names, departed_players(excel_file))
    df, generic_slots = collapse_fungible_bench(df, current_team_names, fixture_columns)
    df = df.reset_index(drop=True)

    # Ownership for the optional two-stage tie-break: when many squads score ~the same XP,
    # break the tie by ownership (differential = favour low-owned, template = favour high).
    if tie_breaker == 'ownership':
        own = load_ownership()
        df['_ownership'] = df['Player Name'].map(own).fillna(0.0)
        print(f"Tie-break: {tie_break_mode} by ownership, within {xp_tolerance:.2f} XP of optimal")

    # Identify current team players in database
    current_team_indices = []
    current_team_cost = 0
    forced_out_indices = []

    for player_name in current_team_names:
        matches = df[df['Player Name'].str.strip() == player_name.strip()]
        if len(matches) == 0:
            matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]

        if len(matches) > 0:
            player_idx = matches.index[0]
            current_team_indices.append(player_idx)
            current_team_cost += df.loc[player_idx, 'Cost']

            # Check if this player is in the forced transfer out list
            if player_name.strip() in [p.strip() for p in force_transfer_out]:
                forced_out_indices.append(player_idx)

    # Validate forced transfers
    if len(forced_out_indices) != len(force_transfer_out):
        missing_players = []
        for forced_name in force_transfer_out:
            found = False
            for current_name in current_team_names:
                if forced_name.strip().lower() == current_name.strip().lower():
                    found = True
                    break
            if not found:
                missing_players.append(forced_name)

        if missing_players:
            print(f"WARNING: The following forced transfer players were not found in current team: {missing_players}")

    # Check if forced transfers exceed max transfers
    num_forced_transfers = len(forced_out_indices)
    if num_forced_transfers > max_transfers:
        print(f"ERROR: Number of forced transfers ({num_forced_transfers}) exceeds max transfers ({max_transfers})")
        return None

    # Resolve forced transfers IN: players the squad MUST contain (a new pickup to test, or an
    # owned player protected from sale). Matched against the finalised pool, so a departed or
    # collapsed name will not resolve. Budget/position limits still apply, so an unaffordable
    # pick simply makes the problem infeasible (no solution) rather than being smuggled in.
    forced_in_indices = []
    for forced_name in force_transfer_in:
        matches = df[df['Player Name'].str.strip() == forced_name.strip()]
        if len(matches) == 0:
            matches = df[df['Player Name'].str.contains(forced_name.strip(), case=False, na=False)]
        if len(matches) > 0:
            forced_in_indices.append(matches.index[0])
        else:
            print(f"WARNING: forced-in player not found in pool (departed/collapsed/typo?): {forced_name}")

    # Calculate total budget available
    total_budget = current_team_cost + additional_budget

    # Calculate current team points for comparison
    current_team_df, current_points, _ = analyse_current_team(excel_file, current_team_names,
                                                              num_fixtures, fixture_weights,
                                                              players_sheet, additional_budget)

    # Store all solutions
    all_solutions = []
    excluded_transfer_combinations = []
    max_xp = None    # XP ceiling for the two-stage tie-break, found once on the first solve

    # Solve multiple times to get top N solutions
    for solution_num in range(num_solutions):
        print(f"\nSolving for solution #{solution_num + 1}...")

        # Create optimisation problem
        prob = pulp.LpProblem(f"FPL_Transfer_Optimisation_{solution_num}", pulp.LpMaximize)

        # Decision variables
        # 1. Squad selection: binary variable for each player in 15-man squad
        squad_vars = {}
        for i in df.index:
            squad_vars[i] = pulp.LpVariable(f"squad_{i}_{solution_num}", cat='Binary')

        # 2. Weekly starting XI: binary variable for each player starting in each fixture
        starting_vars = {}
        for fixture in fixtures:
            starting_vars[fixture] = {}
            for i in df.index:
                starting_vars[fixture][i] = pulp.LpVariable(f"starting_{fixture}_{i}_{solution_num}", cat='Binary')

        # 3. Weekly captains: binary variable for each player being captain in each fixture
        captain_vars = {}
        for fixture in fixtures:
            captain_vars[fixture] = {}
            for i in df.index:
                captain_vars[fixture][i] = pulp.LpVariable(f"captain_{fixture}_{i}_{solution_num}", cat='Binary')

        # 4. Bench players: binary variable for players who are in squad but not starting
        bench_vars = {}
        for fixture in fixtures:
            bench_vars[fixture] = {}
            for i in df.index:
                bench_vars[fixture][i] = pulp.LpVariable(f"bench_{fixture}_{i}_{solution_num}", cat='Binary')

        # transfer_out_vars: 1 if current player is transferred out, 0 otherwise
        transfer_out_vars = {}
        for i in current_team_indices:
            transfer_out_vars[i] = pulp.LpVariable(f"transfer_out_{i}_{solution_num}", cat='Binary')

        # transfer_in_vars: 1 if new player is transferred in, 0 otherwise
        transfer_in_vars = {}
        for i in df.index:
            if i not in current_team_indices:
                transfer_in_vars[i] = pulp.LpVariable(f"transfer_in_{i}_{solution_num}", cat='Binary')

        # Objective function: maximize total weighted expected points from starting XIs + captain bonuses + bench value
        objective_terms = []

        # Add weighted points from starting players in each fixture
        gk_mask = df['Position'] == 'GK'
        outfield_index = [p for p in df.index if not gk_mask[p]]
        for i, fixture in enumerate(fixtures):
            fixture_col = f'{fixture} XP'
            weight = weights[i]
            gk_bench_weight = gk_bench_weights[i]

            for player_idx in df.index:
                # Weighted points from starting players
                objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
                # Weighted captain bonus (additional points for captain)
                objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])

                # GK bench value (outfield bench is priced by sub order below)
                if gk_mask[player_idx]:
                    objective_terms.append(
                        df.loc[player_idx, fixture_col] * weight * gk_bench_weight * bench_vars[fixture][player_idx])

            objective_terms.extend(_bench_slot_objective(
                prob, df[fixture_col], bench_vars[fixture], outfield_index, weight,
                bench_slot_weights[i], f"s{solution_num}_{fixture}"))

        xp_expr = pulp.lpSum(objective_terms)
        # Value preference: charge value_weight weighted-XP per £1.0m of squad cost, so the optimiser
        # keeps money in the bank unless XP pays for the spend. A transfer to a player £0.1m cheaper is
        # then taken even at up to value_weight*0.1 lower weighted XP (value_weight=1.0 -> 0.5 XP for a
        # £0.5m saving, per Tyrone's spec). value_weight=0.0 -> pure XP, unchanged. Folded into xp_expr
        # so the tie-break ceiling/tolerance track the SAME objective; reported points are recomputed
        # from the chosen squad below, so this only steers selection, it does not distort displayed XP.
        if value_weight:
            squad_cost = pulp.lpSum(df.loc[i, 'Cost'] * squad_vars[i] for i in df.index)
            xp_expr = xp_expr - value_weight * squad_cost
        prob += xp_expr

        # Constraint 1: Exactly 15 players in squad
        prob += pulp.lpSum([squad_vars[i] for i in df.index]) == 15

        # Constraint 2: Squad position requirements
        squad_position_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for position, required_count in squad_position_requirements.items():
            position_players = df[df['Position'] == position].index
            prob += pulp.lpSum([squad_vars[i] for i in position_players]) == required_count

        # Constraint 3: Maximum 3 players per team. Generic 0-XP fillers carry no committed
        # team (you would pick a real filler from whichever team has room), so exempt them.
        unique_teams = [t for t in df['Team'].unique() if t != GENERIC_TEAM]
        for team in unique_teams:
            team_players = df[df['Team'] == team].index
            prob += pulp.lpSum([squad_vars[i] for i in team_players]) <= 3

        # NEW Constraint: Maximum defensive players (GK + DEF) per team
        for team in unique_teams:
            defensive_players = df[(df['Team'] == team) & (df['Position'].isin(['GK', 'DEF']))].index
            prob += pulp.lpSum([squad_vars[i] for i in defensive_players]) <= max_defensive_players_per_team

        # Constraint 4: Transfer logic
        # Current players: either stay (squad_vars[i] = 1, transfer_out_vars[i] = 0)
        # or leave (squad_vars[i] = 0, transfer_out_vars[i] = 1)
        for i in current_team_indices:
            prob += squad_vars[i] + transfer_out_vars[i] == 1

        # New players: either not selected (squad_vars[i] = 0, transfer_in_vars[i] = 0)
        # or transferred in (squad_vars[i] = 1, transfer_in_vars[i] = 1)
        for i in df.index:
            if i not in current_team_indices:
                prob += squad_vars[i] == transfer_in_vars.get(i, 0)

        # Constraint 5: Force specific players to be transferred out
        for forced_idx in forced_out_indices:
            prob += transfer_out_vars[forced_idx] == 1

        # Constraint 5b: Force specific players INTO the final squad (a pickup to test, or an
        # owned player protected from sale). The transfer-logic constraints above then set
        # their transfer_in var automatically when they are not already owned.
        for forced_idx in forced_in_indices:
            prob += squad_vars[forced_idx] == 1

        # Constraint 6: Maximum transfers
        total_transfers = []
        total_transfers.extend([transfer_out_vars[i] for i in current_team_indices])
        total_transfers.extend([transfer_in_vars[i] for i in transfer_in_vars])

        prob += pulp.lpSum(total_transfers) <= max_transfers * 2  # Each transfer involves out + in

        # Constraint 7: Equal transfers in and out
        transfers_out = pulp.lpSum([transfer_out_vars[i] for i in current_team_indices])
        transfers_in = pulp.lpSum([transfer_in_vars[i] for i in transfer_in_vars])
        prob += transfers_out == transfers_in

        # Constraint 8: Budget constraint
        cost_terms = []

        # Add costs of players staying in team
        for i in current_team_indices:
            cost_terms.append(df.loc[i, 'Cost'] * (1 - transfer_out_vars[i]))

        # Add costs of new players transferred in
        for i in transfer_in_vars:
            cost_terms.append(df.loc[i, 'Cost'] * transfer_in_vars[i])

        prob += pulp.lpSum(cost_terms) <= total_budget

        # Constraint 9: Starting XI constraints for each fixture
        starting_position_requirements = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}  # Minimum requirements

        for fixture in fixtures:
            # Exactly 11 starters per fixture
            prob += pulp.lpSum([starting_vars[fixture][i] for i in df.index]) == 11

            # Position requirements for starting XI (minimum)
            for position, min_count in starting_position_requirements.items():
                position_players = df[df['Position'] == position].index
                prob += pulp.lpSum([starting_vars[fixture][i] for i in position_players]) >= min_count

            # Maximum position constraints for starting XI
            max_gk = 1
            max_def = 5
            max_mid = 5
            max_fwd = 3

            gk_players = df[df['Position'] == 'GK'].index
            def_players = df[df['Position'] == 'DEF'].index
            mid_players = df[df['Position'] == 'MID'].index
            fwd_players = df[df['Position'] == 'FWD'].index

            prob += pulp.lpSum([starting_vars[fixture][i] for i in gk_players]) <= max_gk
            prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) <= max_def
            prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) <= max_mid
            prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) <= max_fwd

        # Constraint 10: Captain constraints for each fixture
        for fixture in fixtures:
            # Exactly 1 captain per fixture
            prob += pulp.lpSum([captain_vars[fixture][i] for i in df.index]) == 1

            # Captain must be a starter
            for i in df.index:
                prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

        # Constraint 11: Bench constraints for each fixture
        for fixture in fixtures:
            # Exactly 4 bench players per fixture (15 - 11)
            prob += pulp.lpSum([bench_vars[fixture][i] for i in df.index]) == 4

            # Bench players must be in squad but not starting
            for i in df.index:
                prob += bench_vars[fixture][i] <= squad_vars[i]
                prob += bench_vars[fixture][i] <= (1 - starting_vars[fixture][i])
                # If in squad and not starting, must be on bench
                prob += bench_vars[fixture][i] >= squad_vars[i] - starting_vars[fixture][i]

        # Constraint 12: Starting players must be in squad
        for fixture in fixtures:
            for i in df.index:
                prob += starting_vars[fixture][i] <= squad_vars[i]

        # NEW: Constraint 13: Exclude previously found transfer combinations
        for prev_combination in excluded_transfer_combinations:
            out_indices = prev_combination['out']
            in_indices = prev_combination['in']

            # A 0-transfer solution has no combination to ban - force at least one
            # transfer instead, so later solutions surface the next-best alternatives
            if not out_indices and not in_indices:
                prob += pulp.lpSum([transfer_out_vars[i] for i in current_team_indices]) >= 1
                continue

            # This combination cannot be used again
            # Sum of all matching transfer_out and transfer_in must be < total transfers
            matching_transfers = []
            for idx in out_indices:
                if idx in transfer_out_vars:
                    matching_transfers.append(transfer_out_vars[idx])
            for idx in in_indices:
                if idx in transfer_in_vars:
                    matching_transfers.append(transfer_in_vars[idx])

            # If all these transfers match, sum would equal len(matching_transfers)
            # We prevent this by requiring sum < len(matching_transfers)
            if matching_transfers:
                prob += pulp.lpSum(matching_transfers) <= len(matching_transfers) - 1

        # Two-stage tie-break (optional): find the XP ceiling once, then among squads within
        # xp_tolerance of it, maximise ownership (differential = favour low-owned, template =
        # high-owned). This shapes the SQUAD / transfers only — the XI you field is ALWAYS set
        # afterwards by pure XP (pure_xp_lineup), never by ownership.
        if tie_breaker == 'ownership':
            if max_xp is None:
                prob.solve(pulp.PULP_CBC_CMD(msg=0))          # stage 1: the pure-XP ceiling
                if prob.status == pulp.LpStatusOptimal:
                    max_xp = pulp.value(xp_expr)
            if max_xp is not None:
                sign = -1.0 if tie_break_mode == 'differential' else 1.0
                f1 = fixtures[0]
                secondary = pulp.lpSum(sign * float(df.loc[i, '_ownership']) * starting_vars[f1][i]
                                       for i in df.index)
                # Keep what you already own when it doesn't cost XP. This settles the fungible
                # slots the ownership tilt can't see — above all the BENCH KEEPER: a benched GK is
                # worth ~nothing, so without this the optimiser freely swaps your owned Palmer for
                # another equivalent keeper. Ranked below ownership (0.01 vs ~1/point), above XP.
                keep_owned = pulp.lpSum(squad_vars[i] for i in current_team_indices)
                prob += xp_expr >= max_xp - xp_tolerance      # stay within tolerance of optimal
                prob += secondary + 0.01 * keep_owned + 1e-4 * xp_expr   # tilt > keep-owned > XP

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Check solution status
        status = pulp.LpStatus[prob.status]
        print(f"Solution #{solution_num + 1} Status: {status}")

        if prob.status != pulp.LpStatusOptimal:
            print(f"No more optimal solutions found after {solution_num} solution(s)")
            break

        # Extract results
        final_squad_indices = []
        for i in df.index:
            if squad_vars[i].varValue == 1:
                final_squad_indices.append(i)

        transfers_out = []
        for i in current_team_indices:
            if transfer_out_vars[i].varValue == 1:
                transfers_out.append(i)

        transfers_in = []
        for i in transfer_in_vars:
            if transfer_in_vars[i].varValue == 1:
                transfers_in.append(i)

        # Add this combination to excluded list
        excluded_transfer_combinations.append({
            'out': transfers_out.copy(),
            'in': transfers_in.copy()
        })

        # Extract starting XI and captains for each fixture
        starting_lineups = {}
        captains = {}
        bench_players = {}

        # The LP's starting_vars can be tilted by the ownership tie-break; that tilt is
        # allowed to shape the SQUAD (transfers) exactly as before. But once the 15 is fixed,
        # ALWAYS set the XI / bench order / captain by pure expected points, so who you field
        # is never affected by ownership. (Tyrone, 2026-08-21.)
        for fixture in fixtures:
            xi, bench, cap = pure_xp_lineup(final_squad_indices, df, f'{fixture} XP')
            starting_lineups[fixture] = xi
            bench_players[fixture] = bench
            captains[fixture] = cap

        final_squad = df.loc[final_squad_indices].copy()
        transfers_out_players = df.loc[transfers_out].copy() if transfers_out else pd.DataFrame()
        transfers_in_players = df.loc[transfers_in].copy() if transfers_in else pd.DataFrame()

        final_cost = final_squad['Cost'].sum()
        budget_used = final_cost
        budget_remaining = total_budget - budget_used

        # Calculate points for different objective components
        total_starting_points = 0
        total_captain_points = 0
        total_bench_points = 0

        for i, fixture in enumerate(fixtures):
            fixture_col = f'{fixture} XP'
            weight = weights[i]

            # Starting points
            for player_idx in starting_lineups[fixture]:
                total_starting_points += df.loc[player_idx, fixture_col] * weight

            # Captain bonus points
            if fixture in captains:
                captain_idx = captains[fixture]
                total_captain_points += df.loc[captain_idx, fixture_col] * weight

            # Bench points - outfielders by sub order, GK separately
            total_bench_points += weight * bench_points_for_fixture(
                df, bench_players[fixture], fixture_col, bench_slot_weights[i], gk_bench_weights[i])

        final_points = total_starting_points + total_captain_points + total_bench_points

        # Calculate starting XI points without bench
        starting_xi_points = total_starting_points + total_captain_points

        # Calculate F1 (next gameweek) points from optimizer's solution
        f1_fixture = fixtures[0]
        f1_starting_points = 0
        f1_captain_points = 0
        f1_bench_points = 0

        for player_idx in starting_lineups[f1_fixture]:
            f1_starting_points += df.loc[player_idx, 'F1 XP']

        if f1_fixture in captains:
            captain_idx = captains[f1_fixture]
            f1_captain_points += df.loc[captain_idx, 'F1 XP']

        f1_bench_points = bench_points_for_fixture(
            df, bench_players[f1_fixture], 'F1 XP', bench_slot_weights[0], gk_bench_weights[0])

        f1_starting_xi_points = f1_starting_points + f1_captain_points
        f1_total_points = f1_starting_xi_points + f1_bench_points

        num_transfers = len(transfers_out)

        # Calculate optimised baseline using mini-optimizer
        # This properly simulates dynamic lineup selection across fixtures
        baseline_result = calculate_optimised_baseline(
            df, current_team_indices, fixtures, weights, bench_slot_weights, gk_bench_weights
        )

        if baseline_result is None:
            # Fallback to fixture-by-fixture optimization if mini-optimizer fails
            print("Warning: Using fallback baseline calculation")
            from collections import defaultdict

            # Calculate baseline by optimizing each fixture separately
            total_weighted_baseline = 0
            total_starting_baseline = 0
            f1_squad_baseline = 0
            f1_starting_baseline = 0

            for fix_idx, fixture in enumerate(fixtures):
                fixture_col = f'{fixture} XP'
                weight = weights[fix_idx]
                gk_bench_weight = gk_bench_weights[fix_idx]

                # Get all current team players with their fixture XP
                current_team_xp = []
                for idx in current_team_indices:
                    current_team_xp.append({
                        'idx': idx,
                        'xp': df.loc[idx, fixture_col],
                        'position': df.loc[idx, 'Position']
                    })

                # Group by position
                position_groups = defaultdict(list)
                for p in current_team_xp:
                    position_groups[p['position']].append(p)

                # Sort each position by XP for this fixture
                for pos in position_groups:
                    position_groups[pos].sort(key=lambda x: x['xp'], reverse=True)

                # Select starting XI: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
                starting_xi = []
                starting_xi.extend(position_groups['GK'][:1])  # 1 GK
                starting_xi.extend(position_groups['DEF'][:3])  # 3 DEF min
                starting_xi.extend(position_groups['MID'][:2])  # 2 MID min
                starting_xi.extend(position_groups['FWD'][:1])  # 1 FWD min

                # Fill remaining 4 spots
                remaining = []
                remaining.extend(position_groups['DEF'][3:5])  # up to 2 more DEF
                remaining.extend(position_groups['MID'][2:5])  # up to 3 more MID
                remaining.extend(position_groups['FWD'][1:3])  # up to 2 more FWD
                remaining.sort(key=lambda x: x['xp'], reverse=True)
                starting_xi.extend(remaining[:4])

                # Calculate points for this fixture
                starting_points = sum([p['xp'] for p in starting_xi])

                # Captain = best player
                all_sorted = sorted(current_team_xp, key=lambda x: x['xp'], reverse=True)
                captain_bonus = all_sorted[0]['xp'] if len(all_sorted) > 0 else 0

                # Bench - outfielders by sub order, GK separately
                bench_idx = [p['idx'] for p in current_team_xp if p not in starting_xi]
                bench_points = bench_points_for_fixture(
                    df, bench_idx, fixture_col, bench_slot_weights[fix_idx], gk_bench_weight)

                # Add to totals (weighted)
                fixture_starting_total = starting_points + captain_bonus
                fixture_squad_total = fixture_starting_total + bench_points

                total_starting_baseline += fixture_starting_total * weight
                total_weighted_baseline += fixture_squad_total * weight

                # Save F1 specifically
                if fix_idx == 0:
                    f1_starting_baseline = fixture_starting_total
                    f1_squad_baseline = fixture_squad_total

            current_total_with_bench_weight = total_weighted_baseline
            current_starting_weighted = total_starting_baseline
            current_f1_squad_total = f1_squad_baseline
            current_f1_starting_xi = f1_starting_baseline
        else:
            # Use optimised baseline results
            current_total_with_bench_weight, current_starting_weighted, current_f1_squad_total, current_f1_starting_xi = baseline_result

        # Calculate improvements
        points_improvement = final_points - current_total_with_bench_weight
        starting_xi_improvement = starting_xi_points - current_starting_weighted

        f1_squad_improvement = f1_total_points - current_f1_squad_total
        f1_starting_improvement = f1_starting_xi_points - current_f1_starting_xi

        # Fix: For 0 transfers, improvements should be 0
        if num_transfers == 0:
            points_improvement = 0.0
            starting_xi_improvement = 0.0
            f1_squad_improvement = 0.0
            f1_starting_improvement = 0.0

        # Store F1 breakdown for display
        f1_breakdown = {
            'current_f1_starting_xi': current_f1_starting_xi,
            'f1_starting_xi_points': f1_starting_xi_points,
        }

        solution_result = {
            'solution_number': solution_num + 1,
            'final_squad': final_squad,
            'transfers_out': transfers_out_players,
            'transfers_in': transfers_in_players,
            'starting_lineups': starting_lineups,
            'captains': captains,
            'bench_players': bench_players,
            'final_cost': final_cost,
            'total_budget': total_budget,
            'budget_remaining': budget_remaining,
            'final_points': final_points,
            'starting_xi_points': starting_xi_points,
            'total_starting_points': total_starting_points,
            'total_captain_points': total_captain_points,
            'total_bench_points': total_bench_points,
            'current_points': current_total_with_bench_weight,
            'current_starting_weighted': current_starting_weighted,
            'current_total_with_bench_weight': current_total_with_bench_weight,
            'points_improvement': points_improvement,
            'starting_xi_improvement': starting_xi_improvement,
            'f1_total_points': f1_total_points,
            'f1_starting_xi_points': f1_starting_xi_points,
            'current_f1_squad_total': current_f1_squad_total,
            'current_f1_starting_xi': current_f1_starting_xi,
            'f1_squad_improvement': f1_squad_improvement,
            'f1_starting_improvement': f1_starting_improvement,
            'f1_breakdown': f1_breakdown,
            'num_transfers': num_transfers,
            'max_transfers': max_transfers,
            'weights': weights,
            'additional_budget': additional_budget,
            'fixtures': fixtures,
            'bench_slot_weights': bench_slot_weights,
            'gk_bench_weights': gk_bench_weights,
            'forced_transfers_out': force_transfer_out,
            'forced_out_indices': forced_out_indices,
            'max_defensive_players_per_team': max_defensive_players_per_team
        }

        all_solutions.append(solution_result)

    # Return the finalised pool alongside the solutions: departed players dropped, fungible
    # fillers collapsed/renamed, index reset. Every display keyed by solution indices MUST use
    # THIS df, or names/indices will not line up (role table blanks out, horizon XI collapses).
    return all_solutions, df


def display_f1_starting_xi_comparison(solution, df, current_team_indices, transfers_out):
    print(f"\n  Detailed F1 Starting XI Comparison for Option {solution['solution_number']}:")
    print("  " + "-" * 76)

    # Get current team staying players
    current_staying_indices = [idx for idx in current_team_indices if idx not in transfers_out]

    # Sort by F1 XP to find current starting XI
    current_f1_values = []
    for idx in current_staying_indices:
        current_f1_values.append({
            'idx': idx,
            'name': df.loc[idx, 'Player Name'],
            'team': df.loc[idx, 'Team'],
            'f1_xp': df.loc[idx, 'F1 XP'],
            'position': df.loc[idx, 'Position']
        })

    current_f1_values.sort(key=lambda x: x['f1_xp'], reverse=True)

    print(f"\n  CURRENT Starting XI (top 11 by F1 XP from remaining {len(current_staying_indices)} players):")
    current_xi_total = 0
    for i, player in enumerate(current_f1_values[:11], 1):
        print(f"    {i:2d}. {player['name']:<25} {player['team']:<12} {player['f1_xp']:>5.2f} pts")
        current_xi_total += player['f1_xp']
    print(f"    {'TOTAL:':<28} {'':<12} {current_xi_total:>5.2f} pts")

    # Show new starting XI
    f1_fixture = solution['fixtures'][0]
    starting_lineup_indices = solution['starting_lineups'][f1_fixture]
    captain_idx = solution['captains'].get(f1_fixture)

    print(f"\n  NEW Starting XI (optimised):")
    new_xi_total = 0
    new_xi_players = []
    for idx in starting_lineup_indices:
        is_captain = (idx == captain_idx)
        captain_bonus = df.loc[idx, 'F1 XP'] if is_captain else 0
        new_xi_players.append({
            'name': df.loc[idx, 'Player Name'],
            'team': df.loc[idx, 'Team'],
            'f1_xp': df.loc[idx, 'F1 XP'],
            'is_captain': is_captain,
            'is_new': idx in solution['transfers_in'].index.tolist() if len(solution['transfers_in']) > 0 else False
        })
        new_xi_total += df.loc[idx, 'F1 XP']
        if is_captain:
            new_xi_total += captain_bonus

    new_xi_players.sort(key=lambda x: x['f1_xp'], reverse=True)

    for i, player in enumerate(new_xi_players, 1):
        captain_mark = " (C)" if player['is_captain'] else ""
        new_mark = " *NEW*" if player['is_new'] else ""
        print(
            f"    {i:2d}. {player['name']:<25} {player['team']:<12} {player['f1_xp']:>5.2f} pts{captain_mark}{new_mark}")
    print(f"    {'TOTAL:':<28} {'':<12} {new_xi_total:>5.2f} pts")

    print(f"\n  Improvement: +{new_xi_total - current_xi_total:.2f} pts")
    print("  " + "-" * 76)


def load_ownership():
    """Global FPL ownership % per player, keyed by our canonical names.

    Not in the master (it is not a projection input), so it is read straight from the FPL
    snapshot. Best-effort: an empty mapping just leaves the column blank rather than
    breaking a transfer run.
    """
    try:
        from fpl_pipeline import config, names
        path = os.path.join(config.FPL_DATA_DIR, "playerstats.csv")
        keep = {"id", "gw", "first_name", "second_name", "selected_by_percent"}
        d = pd.read_csv(path, usecols=lambda c: c in keep)
        if "gw" in d.columns:
            d = d.sort_values("gw").drop_duplicates(subset="id", keep="last")
        full = names.apply_player_names(d["first_name"] + " " + d["second_name"])
        return dict(zip(full, d["selected_by_percent"]))
    except Exception as exc:
        print(f"  (ownership unavailable: {exc})")
        return {}


def display_role_frequency(solution, df, fixtures):
    """For ONE solution: across the fixture horizon, how many gameweeks each player
    spends starting / 1st sub / 2nd sub / 3rd sub (backup GK shown separately).

    Bench sub order follows the optimiser's own convention in bench_points_for_fixture:
    among benched outfielders, the highest projected XP for that fixture is the 1st sub
    (most likely to actually earn points via an auto-substitution), and so on. So a player
    who is "1st sub" in 5 of 8 gameweeks is the squad's most valuable non-guaranteed body.
    """
    n = len(fixtures)
    roles = {}   # name -> [starts, sub1, sub2, sub3, gk_bench]
    meta = {}
    for _, p in solution['final_squad'].iterrows():
        roles[p['Player Name']] = [0, 0, 0, 0, 0]
        meta[p['Player Name']] = (p['Position'], p['Team'], p['Cost'])
    idx_name = dict(zip(df.index, df['Player Name']))

    for fx in fixtures:
        starters = set(solution['starting_lineups'].get(fx, []))
        bench = solution['bench_players'].get(fx, [])
        for i in starters:
            nm = idx_name.get(i)
            if nm in roles:
                roles[nm][0] += 1
        gk_bench = [i for i in bench if df.loc[i, 'Position'] == 'GK']
        out_bench = sorted((i for i in bench if df.loc[i, 'Position'] != 'GK'),
                           key=lambda i: df.loc[i, f'{fx} XP'], reverse=True)
        for slot, i in enumerate(out_bench[:3]):
            nm = idx_name.get(i)
            if nm in roles:
                roles[nm][slot + 1] += 1
        for i in gk_bench:
            nm = idx_name.get(i)
            if nm in roles:
                roles[nm][4] += 1

    print(f"\n  ROLE ACROSS {n} FIXTURES (of {n} gameweeks, how many as each role)")
    print(f"  {'Player':<26}{'Pos':<5}{'Team':<12}{'Start':>6}{'Sub1':>6}{'Sub2':>6}"
          f"{'Sub3':>6}{'GKbn':>6}")
    order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    for nm in sorted(roles, key=lambda k: (order.get(meta[k][0], 9), -roles[k][0])):
        pos, team, cost = meta[nm]
        st, s1, s2, s3, gb = roles[nm]
        cells = "".join(f"{v:>6}" if v else f"{'-':>6}" for v in (st, s1, s2, s3, gb))
        print(f"  {nm:<26}{pos:<5}{team:<12}{cells}")


def display_squad_frequency(all_solutions, fixtures, weights, current_team_names=None):
    """Every player appearing in ANY near-optimal squad, by position, with how often.

    The transfer-frequency view answers "what should I change?"; this answers "who is in
    the squad regardless?". A player in 20/20 solutions is one the optimiser will not give
    up whatever else it does — worth more confidence than a 3/20 player who only appears
    when some other pick goes a particular way.
    """
    n = len(all_solutions)
    if n < 2:
        return

    seen, owned = {}, set(current_team_names or [])
    for solution in all_solutions:
        for _, p in solution['final_squad'].iterrows():
            name = p['Player Name']
            if name not in seen:
                seen[name] = {'count': 0, 'team': p['Team'], 'cost': p['Cost'],
                              'position': p['Position'], 'f1_xp': p['F1 XP'],
                              'weighted_xp': sum(p[f'{f} XP'] * w for f, w in zip(fixtures, weights))}
            seen[name]['count'] += 1

    ownership = load_ownership()

    print("\n" + "=" * 118)
    print(f"SQUAD SELECTION FREQUENCY — across {n} near-optimal solutions")
    print("=" * 118)
    print(f"{'Player':<28} {'Team':<12} {'Cost':<7} {'F1 XP':<8} {'Wtd XP':<9} "
          f"{'Picked':<10} {'%':<6} {'Mine':<6} {'FPL own%':<9}")

    for position in ('GK', 'DEF', 'MID', 'FWD'):
        rows = sorted(((k, v) for k, v in seen.items() if v['position'] == position),
                      key=lambda kv: (-kv[1]['count'], -kv[1]['weighted_xp']))
        if not rows:
            continue
        print(f"\n{position} ({len(rows)} distinct players across the pool)")
        print("-" * 118)
        for name, d in rows:
            picked = f"{d['count']}/{n}"
            pct = d['count'] / n * 100
            own = ownership.get(name)
            own_str = f"{own:.1f}%" if own is not None else "?"
            print(f"{name:<28} {d['team']:<12} £{d['cost']:<6.1f} {d['f1_xp']:<8.2f} "
                  f"{d['weighted_xp']:<9.2f} {picked:<10} "
                  f"{pct:<5.0f}% {'yes' if name in owned else '':<6} {own_str:<9}")

    locks = sorted(k for k, v in seen.items() if v['count'] == n)
    marginal = sorted((k for k, v in seen.items() if v['count'] <= max(1, n // 5)),
                      key=lambda k: seen[k]['position'])
    print("\n" + "-" * 110)
    print(f"In EVERY solution ({len(locks)}): {', '.join(locks) if locks else 'none'}")
    if marginal:
        print(f"Marginal, in <=20% ({len(marginal)}): {', '.join(marginal)}")


def analyse_transfer_frequency(all_solutions, fixtures, weights):
    transfers_in_count = {}
    transfers_out_count = {}

    for solution in all_solutions:
        # Count transfers out
        for _, player in solution['transfers_out'].iterrows():
            player_name = player['Player Name']
            if player_name not in transfers_out_count:
                # Calculate weighted expected points
                weighted_xp = 0
                for i, fixture in enumerate(fixtures):
                    weighted_xp += player[f'{fixture} XP'] * weights[i]

                transfers_out_count[player_name] = {
                    'count': 0,
                    'team': player['Team'],
                    'cost': player['Cost'],
                    'position': player['Position'],
                    'f1_xp': player['F1 XP'],
                    'weighted_xp': weighted_xp
                }
            transfers_out_count[player_name]['count'] += 1

        # Count transfers in
        for _, player in solution['transfers_in'].iterrows():
            player_name = player['Player Name']
            if player_name not in transfers_in_count:
                # Calculate weighted expected points
                weighted_xp = 0
                for i, fixture in enumerate(fixtures):
                    weighted_xp += player[f'{fixture} XP'] * weights[i]

                transfers_in_count[player_name] = {
                    'count': 0,
                    'team': player['Team'],
                    'cost': player['Cost'],
                    'position': player['Position'],
                    'f1_xp': player['F1 XP'],
                    'weighted_xp': weighted_xp
                }
            transfers_in_count[player_name]['count'] += 1

    # Sort by frequency
    sorted_out = sorted(transfers_out_count.items(), key=lambda x: x[1]['count'], reverse=True)
    sorted_in = sorted(transfers_in_count.items(), key=lambda x: x[1]['count'], reverse=True)

    return {
        'transfers_out': sorted_out,
        'transfers_in': sorted_in,
        'total_solutions': len(all_solutions)
    }


def display_transfer_frequency(frequency_analysis, min_frequency=2):
    total_solutions = frequency_analysis['total_solutions']

    print("\n" + "=" * 110)
    print("TRANSFER FREQUENCY ANALYSIS")
    print("=" * 110)
    print(f"Based on {total_solutions} optimised solutions\n")

    # Most commonly transferred OUT
    print("MOST COMMONLY TRANSFERRED OUT:")
    print("-" * 110)
    print(
        f"{'Player':<30} {'Team':<12} {'Pos':<5} {'Cost':<8} {'F1 XP':<8} {'Total XP':<10} {'Frequency':<12} {'%':<8}")
    print("-" * 110)

    has_common_out = False
    for player_name, data in frequency_analysis['transfers_out']:
        if data['count'] >= min_frequency:
            has_common_out = True
            frequency_pct = (data['count'] / total_solutions) * 100
            print(f"{player_name:<30} {data['team']:<12} {data['position']:<5} "
                  f"£{data['cost']:.1f}m{'':<3} {data['f1_xp']:.1f}{'':<6} "
                  f"{data['weighted_xp']:.1f}{'':<8} "
                  f"{data['count']}/{total_solutions}{'':<6} {frequency_pct:.0f}%")

    if not has_common_out:
        print(f"No players transferred out in {min_frequency}+ solutions")

    # Most commonly transferred IN
    print("\n\nMOST COMMONLY TRANSFERRED IN:")
    print("-" * 110)
    print(
        f"{'Player':<30} {'Team':<12} {'Pos':<5} {'Cost':<8} {'F1 XP':<8} {'Total XP':<10} {'Frequency':<12} {'%':<8}")
    print("-" * 110)

    has_common_in = False
    for player_name, data in frequency_analysis['transfers_in']:
        if data['count'] >= min_frequency:
            has_common_in = True
            frequency_pct = (data['count'] / total_solutions) * 100
            print(f"{player_name:<30} {data['team']:<12} {data['position']:<5} "
                  f"£{data['cost']:.1f}m{'':<3} {data['f1_xp']:.1f}{'':<6} "
                  f"{data['weighted_xp']:.1f}{'':<8} "
                  f"{data['count']}/{total_solutions}{'':<6} {frequency_pct:.0f}%")

    if not has_common_in:
        print(f"No players transferred in {min_frequency}+ solutions")

    # Key insights
    print("\n\nKEY INSIGHTS:")
    print("-" * 110)

    consensus_out = [name for name, data in frequency_analysis['transfers_out']
                     if data['count'] / total_solutions > 0.2]
    consensus_in = [name for name, data in frequency_analysis['transfers_in']
                    if data['count'] / total_solutions > 0.2]

    if consensus_out:
        print(f"Consensus transfers OUT (>20%): {', '.join(consensus_out)}")
    else:
        print("No consensus transfers out (none appear in >20% of solutions)")

    if consensus_in:
        print(f"Consensus transfers IN (>20%): {', '.join(consensus_in)}")
    else:
        print("No consensus transfers in (none appear in >20% of solutions)")


def horizon_bands(solution, df):
    """Raw (unweighted) XI and bench XP, split into near / mid / far fixture bands.

    Deliberately UNWEIGHTED, unlike the optimiser's objective: this answers "what does
    this squad actually score, and when?", so the bands are comparable across solutions
    and readable as points. The weighted figure remains 'Total Starting Points' above.

    XI includes the captain bonus (it is real return). Bench is the plain sum of the four
    subs, i.e. what a Bench Boost on that fixture would be worth.
    """
    fixtures = solution['fixtures']
    bands = [("F1", [fixtures[0]]),
             ("F2-F5", fixtures[1:5]),
             (f"F6-F{len(fixtures)}", fixtures[5:])]
    rows = []
    for label, block in bands:
        if not block:
            continue
        xi = bench = 0.0
        for f in block:
            col = f"{f} XP"
            xi += sum(df.loc[i, col] for i in solution['starting_lineups'][f])
            cap = solution['captains'].get(f)
            if cap is not None:
                xi += df.loc[cap, col]
            bench += sum(df.loc[i, col] for i in solution['bench_players'][f])
        rows.append((label, xi, bench))
    rows.append(("TOTAL", sum(r[1] for r in rows), sum(r[2] for r in rows)))
    return rows


def display_horizon_table(all_solutions, df):
    """One row per computed solution, XI and bench XP split by fixture band.

    Covers EVERY solution in the pool, not just the few displayed in detail — the whole
    point of a compact table is that the also-rans are cheap to show, and a solution
    ranked 9th on the weighted objective may still have the shape you want (a strong
    opening gameweek, say, or a bench worth boosting).
    """
    if not all_solutions or df is None:
        return
    labels = [lab for lab, _, _ in horizon_bands(all_solutions[0], df)]

    print("\n" + "=" * 112)
    print(f"XP BY HORIZON — all {len(all_solutions)} computed solutions, raw and unweighted")
    print("=" * 112)
    head = f"{'Opt':>4}{'Tr':>4}{'Bank':>7}"
    for lab in labels:
        head += f"{lab + ' XI':>11}{lab + ' Bn':>10}"
    print(head + f"{'Weighted':>11}")
    print("-" * 112)

    for s in all_solutions:
        row = f"{s['solution_number']:>4}{s['num_transfers']:>4}{s['budget_remaining']:>6.1f}m"
        for _, xi, bench in horizon_bands(s, df):
            row += f"{xi:>11.2f}{bench:>10.2f}"
        # The RANKING objective is the full weighted total (starting XI + captain + bench),
        # not the starting XI alone — print that so the column is monotonic and Option 1 (the
        # objective maximum) reads highest. total_starting_points omits captain+bench, which is
        # exactly where a top solution can bank its edge (e.g. a stronger bench).
        print(row + f"{s['final_points']:>11.2f}")

    print("-" * 112)
    print("  XI includes the captain bonus; Bench is the plain sum of all four subs, so the")
    print("  bench column doubles as what a Bench Boost on that band would be worth.")
    print("  'Weighted' is the optimiser's actual objective — the ranking column.")


def display_multi_solution_summary(all_solutions, show_f1_breakdown=True, show_detailed_f1=False,
                                   df=None, current_team_indices=None):
    if not all_solutions:
        print("No solutions to display!")
        return

    print("\n" + "=" * 100)
    print("TRANSFER OPTIONS SUMMARY")
    print("=" * 100)

    for solution in all_solutions:
        squad_improvement_sign = "+" if solution['points_improvement'] >= 0 else ""
        starting_improvement_sign = "+" if solution['starting_xi_improvement'] >= 0 else ""
        f1_squad_sign = "+" if solution['f1_squad_improvement'] >= 0 else ""
        f1_starting_sign = "+" if solution['f1_starting_improvement'] >= 0 else ""

        print(f"\nOPTION {solution['solution_number']}:")
        print(
            f"  Total Starting Points: {solution['total_starting_points']:.2f} pts")
        print(
            f"  Total Squad Improvement (with Bench): {squad_improvement_sign}{solution['points_improvement']:.2f} pts")
        print(
            f"  Total Starting XI Improvement: {starting_improvement_sign}{solution['starting_xi_improvement']:.2f} pts")
        print(f"  Next GW Squad Improvement: {f1_squad_sign}{solution['f1_squad_improvement']:.2f} pts")
        print(f"  Next GW Starting XI Improvement: {f1_starting_sign}{solution['f1_starting_improvement']:.2f} pts")
        print(f"  Transfers: {solution['num_transfers']} | Budget Remaining: £{solution['budget_remaining']:.1f}m")

        if len(solution['transfers_out']) > 0:
            print(f"  Transfers Out:")
            for _, player in solution['transfers_out'].iterrows():
                print(
                    f"    - {player['Player Name']:<25} ({player['Team']}, £{player['Cost']:.1f}m, "
                    f"F1: {player['F1 XP']:.1f} pts, Total: {player['Weighted_Total_XP']:.1f} pts)")

        if len(solution['transfers_in']) > 0:
            print(f"  Transfers In:")
            for _, player in solution['transfers_in'].iterrows():
                print(
                    f"    + {player['Player Name']:<25} ({player['Team']}, £{player['Cost']:.1f}m, "
                    f"F1: {player['F1 XP']:.1f} pts, Total: {player['Weighted_Total_XP']:.1f} pts)")

        # Show F1 breakdown if requested
        if show_f1_breakdown and 'f1_breakdown' in solution:
            breakdown = solution['f1_breakdown']
            print(f"\n  F1 Starting XI Breakdown:")
            print(f"    Current Starting XI Total: {breakdown['current_f1_starting_xi']:.2f} pts")
            print(f"    New Starting XI Total: {breakdown['f1_starting_xi_points']:.2f} pts")

        # Show detailed F1 comparison if requested
        if show_detailed_f1 and df is not None and current_team_indices is not None:
            transfers_out_indices = solution['transfers_out'].index.tolist() if len(
                solution['transfers_out']) > 0 else []
            display_f1_starting_xi_comparison(solution, df, current_team_indices, transfers_out_indices)


def display_solution_detail(solution, excel_file, current_team_names):
    print(f"\n" + "=" * 100)
    print(f"DETAILED VIEW - OPTION {solution['solution_number']}")
    print("=" * 100)

    # Show complete final squad
    print(f"\nFINAL SQUAD:")
    print("-" * 80)
    final_squad_sorted = solution['final_squad'].sort_values(['Position', 'Weighted_Total_XP'],
                                                             ascending=[True, False])

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = final_squad_sorted[final_squad_sorted['Position'] == pos]
        if len(pos_players) > 0:
            print(f"\n{pos} ({len(pos_players)}):")
            for _, player in pos_players.iterrows():
                print(f"  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                      f"{player['Weighted_Total_XP']:.2f} pts")

    # Show transfers
    if len(solution['transfers_out']) > 0:
        print(f"\nTRANSFERS OUT ({len(solution['transfers_out'])}):")
        print("-" * 80)
        for _, player in solution['transfers_out'].iterrows():
            print(f"  OUT: {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                  f"{player['Weighted_Total_XP']:.2f} pts")

    if len(solution['transfers_in']) > 0:
        print(f"\nTRANSFERS IN ({len(solution['transfers_in'])}):")
        print("-" * 80)
        for _, player in solution['transfers_in'].iterrows():
            print(f"  IN:  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                  f"{player['Weighted_Total_XP']:.2f} pts")

    # Show starting XI for next fixture
    display_starting_lineup_from_solution(solution, fixture_num=1)


def display_starting_lineup_from_solution(solution, fixture_num=1):
    fixtures = solution['fixtures']
    if fixture_num > len(fixtures):
        print(f"Fixture {fixture_num} not available. Only {len(fixtures)} fixtures analyzed.")
        return

    fixture = fixtures[fixture_num - 1]
    starting_lineup = solution['starting_lineups'][fixture]
    captain_idx = solution['captains'].get(fixture)
    bench = solution['bench_players'][fixture]

    final_squad = solution['final_squad']

    print(f"\n" + "=" * 80)
    print(f"OPTIMAL STARTING XI - FIXTURE {fixture_num} ({fixture})")
    print("=" * 80)

    starting_players = final_squad[final_squad.index.isin(starting_lineup)].copy()
    bench_players_data = final_squad[final_squad.index.isin(bench)].copy()

    starting_players['Is_Captain'] = starting_players.index == captain_idx
    # Vice-captain: best remaining starter by this fixture's XP (the armband fallback)
    non_captain = starting_players[~starting_players['Is_Captain']]
    vice_captain_idx = non_captain[f'{fixture} XP'].idxmax() if len(non_captain) else None

    starting_sorted = starting_players.sort_values(['Position', f'{fixture} XP'], ascending=[True, False])

    formation_count = {
        'GK': len(starting_sorted[starting_sorted['Position'] == 'GK']),
        'DEF': len(starting_sorted[starting_sorted['Position'] == 'DEF']),
        'MID': len(starting_sorted[starting_sorted['Position'] == 'MID']),
        'FWD': len(starting_sorted[starting_sorted['Position'] == 'FWD'])
    }

    formation = f"{formation_count['GK']}-{formation_count['DEF']}-{formation_count['MID']}-{formation_count['FWD']}"
    print(f"Formation: {formation}")
    print()

    total_starting_points = 0
    captain_bonus = 0

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = starting_sorted[starting_sorted['Position'] == pos]
        if len(pos_players) > 0:
            print(f"{pos}:")
            for idx, player in pos_players.iterrows():
                points = player[f'{fixture} XP']
                cost = player['Cost']
                team = player['Team']
                captain_mark = " (C)" if player['Is_Captain'] else (" (VC)" if idx == vice_captain_idx else "")
                start_prob = f" ({player[f'{fixture} Start']:.0%})" if f'{fixture} Start' in player else ""
                total_starting_points += points
                if player['Is_Captain']:
                    captain_bonus += points

                print(f"  {player['Player Name']:<25} {team:<8} £{cost:.1f}m  {points:.2f} pts{start_prob}{captain_mark}")
            print()

    print("BENCH:")
    gk_bench = bench_players_data[bench_players_data['Position'] == 'GK']
    outfield_bench = bench_players_data[bench_players_data['Position'] != 'GK'] \
        .sort_values(f'{fixture} XP', ascending=False)
    total_bench_value = 0

    for _, player in gk_bench.iterrows():
        points = player[f'{fixture} XP']
        total_bench_value += points
        start_prob = f" ({player[f'{fixture} Start']:.0%})" if f'{fixture} Start' in player else ""
        print(f"  GK  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  {points:.2f} pts{start_prob}")
    for slot, (_, player) in enumerate(outfield_bench.iterrows(), start=1):
        points = player[f'{fixture} XP']
        total_bench_value += points
        start_prob = f" ({player[f'{fixture} Start']:.0%})" if f'{fixture} Start' in player else ""
        print(f"  {slot}.  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  {points:.2f} pts{start_prob}")

    print(f"\nFIXTURE {fixture_num} SUMMARY:")
    print(f"  Starting XI Points: {total_starting_points + captain_bonus:.2f}")
    print(f"  Bench Points: {total_bench_value:.2f}")

    # Chip watch: what the chips would add if played on this fixture
    captain_name = (starting_players.loc[captain_idx, 'Player Name']
                    if captain_idx in starting_players.index else "?")
    print(f"  Chip watch: Bench Boost +{total_bench_value:.2f} pts | "
          f"Triple Captain +{captain_bonus:.2f} pts ({captain_name})")

    # Key probabilities for this fixture (%): goals, assists, clean sheet, defensive
    # contribution. DefCon is position-gated (the DEF/MID column that applies); a 0 or
    # missing value shows as '-'. XI (grouped by position) first, then the four bench.
    _pcols = [("1 Goal", f"{fixture} Score 1+"), ("2 Goals", f"{fixture} Score 2+"),
              ("1 Asst", f"{fixture} Assist"), ("2 Asst", f"{fixture} Assist 2+"),
              ("CleanSh", f"{fixture} Clean Sheet")]
    _dc = {"DEF": f"{fixture} Defensive Contribution - DEF",
           "MID": f"{fixture} Defensive Contribution - MID"}
    if all(c in final_squad.columns for _, c in _pcols):
        def _prob_row(player):
            pos = str(player['Position'])
            vals = [player[c] for _, c in _pcols]
            dc = _dc.get(pos)
            vals.append(player[dc] if dc and dc in final_squad.columns else float('nan'))
            cells = "".join((f"{v * 100:>7.0f}%" if pd.notna(v) and v > 0 else f"{'-':>8}")
                            for v in vals)
            print(f"    {str(player['Player Name'])[:24]:<25}{pos:<5}{cells}")

        print(f"\n  Key probabilities - {fixture} (%):")
        print("    " + f"{'Player':<25}{'Pos':<5}"
              + "".join(f"{lab:>8}" for lab, _ in _pcols) + f"{'DefCon':>8}")
        for _, player in starting_sorted.iterrows():
            _prob_row(player)
        if len(bench_players_data):
            print("    ---- Bench ----")
            for _, player in pd.concat([gk_bench, outfield_bench]).iterrows():
                _prob_row(player)


def main_multi_transfer_optimiser(excel_file="outputs/13_players_master.csv", max_transfers=2, num_fixtures=5,
                                  fixture_weights=None, show_current_analysis=True,
                                  additional_budget=0.0, bench_slot_weights=None, gk_bench_weights=None,
                                  force_transfer_out=None, force_transfer_in=None,
                                  tie_breaker=None, tie_break_mode='differential', xp_tolerance=0.5,
                                  num_solutions_display=3,
                                  show_all_details=False, show_detailed_f1=False,
                                  compute_solutions=20, show_frequency_analysis=True,
                                  min_frequency=2, max_defensive_players_per_team=3,
                                  ownership_weights=None, reliability_weights=None,
                                  bank_lookahead_gws=None, value_weight=0.0):
    # Fixture weights come from the two components unless one is passed explicitly
    # (an explicit fixture_weights still wins, so old call sites keep working).
    derived = combine_fixture_weights(ownership_weights, reliability_weights, num_fixtures)
    if fixture_weights is None:
        fixture_weights = derived
        print(f"Fixture weights derived from ownership x reliability:")
        print(f"  ownership   {[f'{w:.2f}' for w in (ownership_weights or OWNERSHIP_WEIGHTS)[:num_fixtures]]}")
        print(f"  reliability {[f'{w:.2f}' for w in (reliability_weights or RELIABILITY_WEIGHTS)[:num_fixtures]]}")
        print(f"  combined    {[f'{w:.2f}' for w in derived]}")

    if bench_slot_weights is None:
        bench_slot_weights = BENCH_SLOT_WEIGHTS

    if gk_bench_weights is None:
        gk_bench_weights = [0.10, 0.10, 0.08, 0.06, 0.04, 0.02]

    weights = fixture_weights[:num_fixtures]
    gk_bench_weights_used = gk_bench_weights[:num_fixtures]

    print(f"FPL MULTI-TRANSFER OPTIMISER")
    print(f"Max Transfers: {max_transfers}")
    print(f"Fixtures: {num_fixtures}")
    print(f"Computing {compute_solutions} solutions, displaying top {num_solutions_display}")
    print(f"Weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional Budget: £{additional_budget:.1f}m")
    print(f"Bench Slot Weights (sub order 1/2/3): {bench_slot_weights}")
    print(f"GK Bench Weights: {[f'{w:.2f}' for w in gk_bench_weights_used]}")
    print(f"Max Defensive Players (GK+DEF) per Team: {max_defensive_players_per_team}")

    try:
        # Load current team
        current_team_names, gameweek = load_current_team(excel_file)

        if show_current_analysis:
            # Analyze current team
            current_team_df, current_points, analysis = analyse_current_team(
                excel_file, current_team_names, num_fixtures, fixture_weights,
                'Players', additional_budget
            )

            # Display current team analysis
            display_current_team_analysis(current_team_df, analysis, num_fixtures, weights)

        # Optimise transfers - get multiple solutions AND the finalised pool they index into
        # (departed dropped, fungible fillers collapsed, index reset). All displays below use
        # this df; reloading a fresh one would desync indices/names and blank the role table.
        result = optimise_transfers_multi(
            excel_file, current_team_names, max_transfers, num_fixtures,
            fixture_weights, 'Players', additional_budget, bench_slot_weights,
            gk_bench_weights_used, force_transfer_out, compute_solutions, max_defensive_players_per_team,
            force_transfer_in=force_transfer_in, tie_breaker=tie_breaker,
            tie_break_mode=tie_break_mode, xp_tolerance=xp_tolerance, value_weight=value_weight
        )
        all_solutions, df = result if result else (None, None)

        if not all_solutions:
            print("No solutions found!")
            return None

        # Current team indices, resolved against that same finalised pool
        current_team_indices = []
        for player_name in current_team_names:
            matches = df[df['Player Name'].str.strip() == player_name.strip()]
            if len(matches) == 0:
                matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]
            if len(matches) > 0:
                current_team_indices.append(matches.index[0])

        # Show frequency analysis if requested
        if show_frequency_analysis and len(all_solutions) > 1:
            fixtures = all_solutions[0]['fixtures']
            weights = all_solutions[0]['weights']
            frequency_analysis = analyse_transfer_frequency(all_solutions, fixtures, weights)
            display_transfer_frequency(frequency_analysis, min_frequency)
            display_squad_frequency(all_solutions, fixtures, weights, current_team_names)

        # Display summary of top N solutions only
        # Horizon table covers the whole pool; the detailed summaries below are capped
        display_horizon_table(all_solutions, df)

        solutions_to_display = all_solutions[:num_solutions_display]
        display_multi_solution_summary(solutions_to_display, show_f1_breakdown=True,
                                       show_detailed_f1=show_detailed_f1,
                                       df=df, current_team_indices=current_team_indices)

        for sol in solutions_to_display:
            print("\n" + "=" * 78)
            print(f"OPTION {sol['solution_number']} — per-player role over the horizon")
            print("=" * 78)
            display_role_frequency(sol, df, sol['fixtures'])

        # Display details for solutions if requested
        if show_all_details:
            for solution in solutions_to_display:
                display_solution_detail(solution, excel_file, current_team_names)
        else:
            # Just show details for the best solution
            print("\n" + "=" * 100)
            print(
                f"Showing detailed view for OPTION 1 only. Set show_all_details=True to see all {num_solutions_display} options.")
            display_solution_detail(solutions_to_display[0], excel_file, current_team_names)

        # Transfer-timing plan (bank now vs move now) — printed LAST so it's the final takeaway.
        # Starting free transfers = max_transfers (your GW1 FTs); the plan accrues +1 each GW after.
        if bank_lookahead_gws:
            transfer_timing_check(excel_file, current_team_names, bank_lookahead_gws,
                                  additional_budget, max_transfers)

        return all_solutions

    except FileNotFoundError:
        print(f"Error: Could not find {excel_file}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


result = main_multi_transfer_optimiser(
    excel_file="outputs/13_players_master.csv",
    max_transfers=0,
    additional_budget=0.5,
    num_fixtures=8,
    compute_solutions=1,
    num_solutions_display=1,
    bank_lookahead_gws=1,
    # Weighted-XP charged per £1.0m of squad value: prefers cheaper squads/transfers, trading XP for
    # bank at this rate. 0.0 = pure XP (off). e.g. 1.0 -> accept 0.5 lower weighted XP to save £0.5m.
    value_weight=0.5,
    # How fast the squad churns / how fixable a bad far fixture is - independent of forecast quality
    ownership_weights=[1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558],
    # How much the projection is trusted - re-measure from the backtest as the archive grows
    reliability_weights=[1.0, 0.850, 0.616, 0.578, 0.503, 0.498, 0.481, 0.472],
    show_current_analysis=False,
    bench_slot_weights=(0.25, 0.05, 0.01),   # sub order 1/2/3, applied every fixture
    # bench_slot_weights=[(1.0, 1.0, 1.0)] + [(0.30, 0.10, 0.05)]*7,
    gk_bench_weights=[0.01]*8,
    # gk_bench_weights=[1.0] + [0.05]*7,
    max_defensive_players_per_team=2,
    show_all_details=False,
    show_detailed_f1=False,
    # force_transfer_out=["Rodrigo Muniz"],
    # force_transfer_in=["Erling Haaland"],
    # tie_breaker="ownership",
    # tie_break_mode="differential",
    # xp_tolerance=1.0,
)

if __name__ == "__main__":
    pass
