import pandas as pd
import pulp
import numpy as np


def optimise_fpl_team_with_weekly_lineups_and_bench_value(excel_file, num_fixtures=6, sheet_name='Players',
                                                          fixture_weights=None, bench_weight=0.1,
                                                          total_squad_cost=100.0):

    # Validate num_fixtures parameter
    if not isinstance(num_fixtures, int) or num_fixtures < 1 or num_fixtures > 6:
        print("Error: num_fixtures must be an integer between 1 and 6")
        return None, None, None, None, None, None

    # Set default fixture weights if not provided
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.7, 0.7, 0.7]

    # Ensure we have the right number of weights
    if len(fixture_weights) < num_fixtures:
        print(f"Error: Need at least {num_fixtures} fixture weights, but only {len(fixture_weights)} provided")
        return None, None, None, None, None, None

    # Use only the weights we need
    weights = fixture_weights[:num_fixtures]

    print(f"Using fixture weights: {weights}")
    print(f"Using bench weight: {bench_weight} (bench players get {bench_weight * 100:.1f}% weight)")

    # Load data from Excel
    print("Loading data from Excel...")
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    print("Data length before XP filtering:", len(df))

    # df = df[df['Total XP after XP filtering'] > 0]

    print("Data length:", len(df))

    # Clean column names (remove any extra spaces)
    df.columns = df.columns.str.strip()

    # Define fixture columns based on num_fixtures parameter
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP']
    fixture_columns = all_fixture_columns[:num_fixtures]
    fixtures = [f'F{i + 1}' for i in range(num_fixtures)]

    print(f"Optimising for {num_fixtures} fixture(s): {', '.join(fixtures)}")

    # Check if all fixture columns exist
    missing_columns = [col for col in fixture_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None, None, None, None

    # Calculate weighted total XP for selected fixtures
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight
        print(f"  {fixture_col}: weight = {weight}")

    # Also calculate unweighted total for comparison
    df['Base_Total_XP'] = df[fixture_columns].sum(axis=1)

    # Display data info
    print(f"Loaded {len(df)} players")
    print(f"Fixture XP columns found: {fixture_columns}")
    print(f"\nPosition distribution:")
    print(df['Position'].value_counts())
    print(f"\nCost range: {df['Cost'].min()} - {df['Cost'].max()}")
    print(f"Weighted Total XP range: {df['Weighted_Total_XP'].min():.3f} - {df['Weighted_Total_XP'].max():.3f}")
    print(f"Base Total XP range: {df['Base_Total_XP'].min():.3f} - {df['Base_Total_XP'].max():.3f}")

    # Create the optimisation problem
    prob = pulp.LpProblem("FPL_Team_Selection_With_Weekly_Lineups_And_Bench_Value", pulp.LpMaximize)

    # Decision variables
    # 1. Squad selection: binary variable for each player in 15-man squad
    squad_vars = {}
    for i in df.index:
        squad_vars[i] = pulp.LpVariable(f"squad_{i}", cat='Binary')

    # 2. Weekly starting XI: binary variable for each player starting in each fixture
    starting_vars = {}
    for fixture in fixtures:
        starting_vars[fixture] = {}
        for i in df.index:
            starting_vars[fixture][i] = pulp.LpVariable(f"starting_{fixture}_{i}", cat='Binary')

    # 3. Weekly captains: binary variable for each player being captain in each fixture
    captain_vars = {}
    for fixture in fixtures:
        captain_vars[fixture] = {}
        for i in df.index:
            captain_vars[fixture][i] = pulp.LpVariable(f"captain_{fixture}_{i}", cat='Binary')

    # NEW: 4. Bench players: binary variable for players who are in squad but not starting
    bench_vars = {}
    for fixture in fixtures:
        bench_vars[fixture] = {}
        for i in df.index:
            bench_vars[fixture][i] = pulp.LpVariable(f"bench_{fixture}_{i}", cat='Binary')

    # Objective function: maximize total weighted expected points from starting XIs + captain bonuses + bench value
    objective_terms = []

    # Add weighted points from starting players in each fixture
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        for player_idx in df.index:
            # Weighted points from starting players
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            # Weighted captain bonus (additional points for captain)
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # NEW: Weighted bench value (smaller weight for bench players)
            objective_terms.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

    prob += pulp.lpSum(objective_terms)

    # SQUAD CONSTRAINTS (same as before)

    # Constraint 1: Exactly 15 players in squad
    prob += pulp.lpSum([squad_vars[i] for i in df.index]) == 15

    # Constraint 2: Squad total cost <= total_squad_cost
    prob += pulp.lpSum([df.loc[i, 'Cost'] * squad_vars[i] for i in df.index]) <= total_squad_cost

    # Constraint 3: Squad position requirements (2 GK, 5 DEF, 5 MID, 3 FWD)
    squad_position_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for position, required_count in squad_position_requirements.items():
        position_players = df[df['Position'] == position].index
        prob += pulp.lpSum([squad_vars[i] for i in position_players]) == required_count

    # Constraint 4: Maximum 3 players per team in squad
    unique_teams = df['Team'].unique()
    for team in unique_teams:
        team_players = df[df['Team'] == team].index
        prob += pulp.lpSum([squad_vars[i] for i in team_players]) <= 3

    # Constraint 5: Budget GK strategy (at least one GK <= 4.0)
    # budget_gks = df[(df['Position'] == 'GK') & (df['Cost'] <= 4.0)].index
    # if len(budget_gks) > 0:
    #     prob += pulp.lpSum([squad_vars[i] for i in budget_gks]) >= 1

    # WEEKLY LINEUP CONSTRAINTS (same as before)

    for fixture in fixtures:
        # Constraint 6: Exactly 11 starters per fixture
        prob += pulp.lpSum([starting_vars[fixture][i] for i in df.index]) == 11

        # Constraint 7: Can only start players who are in the squad
        for i in df.index:
            prob += starting_vars[fixture][i] <= squad_vars[i]

        # Constraint 8: Weekly formation constraints
        # Exactly 1 GK
        gk_players = df[df['Position'] == 'GK'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in gk_players]) == 1

        # 3-5 DEF
        def_players = df[df['Position'] == 'DEF'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) <= 5

        # 3-5 MID
        mid_players = df[df['Position'] == 'MID'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) <= 5

        # 1-3 FWD
        fwd_players = df[df['Position'] == 'FWD'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) >= 1
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) <= 3

    # CAPTAIN CONSTRAINTS (same as before)

    for fixture in fixtures:
        # Constraint 9: Exactly one captain per fixture
        prob += pulp.lpSum([captain_vars[fixture][i] for i in df.index]) == 1

        # Constraint 10: Captain must be a starter
        for i in df.index:
            prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

    # NEW: BENCH CONSTRAINTS

    for fixture in fixtures:
        for i in df.index:
            # Constraint 11: Bench players are in squad but not starting
            prob += bench_vars[fixture][i] <= squad_vars[i]  # Must be in squad to be on bench
            prob += bench_vars[fixture][i] <= 1 - starting_vars[fixture][i]  # Can't be both starting and bench

            # Constraint 12: If in squad and not starting, must be on bench
            prob += bench_vars[fixture][i] >= squad_vars[i] - starting_vars[fixture][i]

    # Solve the problem
    print("\nSolving complex optimisation with weekly lineups, captain rotation, fixture weighting, and bench value...")
    print("This may take longer due to increased complexity...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Check if solution was found
    status = pulp.LpStatus[prob.status]
    print(f"Optimisation Status: {status}")

    if prob.status != pulp.LpStatusOptimal:
        print("No optimal solution found!")
        return None, None, None, None, None, None, None

    # Extract results

    # Squad selection
    squad_indices = []
    for i in df.index:
        if squad_vars[i].varValue == 1:
            squad_indices.append(i)
    squad = df.loc[squad_indices].copy()

    # Weekly starting XIs
    weekly_lineups = {}
    for fixture in fixtures:
        starting_indices = []
        for i in df.index:
            if starting_vars[fixture][i].varValue == 1:
                starting_indices.append(i)
        weekly_lineups[fixture] = df.loc[starting_indices].copy()

    # Weekly captains
    weekly_captains = {}
    for fixture in fixtures:
        for i in df.index:
            if captain_vars[fixture][i].varValue == 1:
                weekly_captains[fixture] = df.loc[i]
                break

    # NEW: Weekly bench players
    weekly_benches = {}
    for fixture in fixtures:
        bench_indices = []
        for i in df.index:
            if bench_vars[fixture][i].varValue == 1:
                bench_indices.append(i)
        weekly_benches[fixture] = df.loc[bench_indices].copy()

    # Calculate totals (both weighted and unweighted)
    squad_cost = squad['Cost'].sum()

    total_weighted_points = 0
    total_unweighted_points = 0
    total_captain_bonus_weighted = 0
    total_captain_bonus_unweighted = 0
    total_bench_value_weighted = 0
    total_bench_value_unweighted = 0

    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]

        # Points from starters (unweighted)
        fixture_points = weekly_lineups[fixture][fixture_col].sum()
        total_unweighted_points += fixture_points

        # Points from starters (weighted)
        weighted_fixture_points = fixture_points * weight
        total_weighted_points += weighted_fixture_points

        # Captain bonus
        if fixture in weekly_captains:
            captain_bonus = weekly_captains[fixture][fixture_col]
            total_unweighted_points += captain_bonus
            total_captain_bonus_unweighted += captain_bonus

            weighted_captain_bonus = captain_bonus * weight
            total_weighted_points += weighted_captain_bonus
            total_captain_bonus_weighted += weighted_captain_bonus

        # NEW: Bench value (for tracking, not added to main total)
        if fixture in weekly_benches:
            bench_value = weekly_benches[fixture][fixture_col].sum()
            total_bench_value_unweighted += bench_value
            total_bench_value_weighted += bench_value * weight

    return (squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
            total_weighted_points, total_captain_bonus_weighted,
            total_unweighted_points, total_captain_bonus_unweighted,
            total_bench_value_weighted, total_bench_value_unweighted, weights)


def display_weighted_lineup_results_with_bench(squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
                                               total_weighted_points, total_captain_bonus_weighted,
                                               total_unweighted_points, total_captain_bonus_unweighted,
                                               total_bench_value_weighted, total_bench_value_unweighted,
                                               weights, num_fixtures=None, total_squad_cost=100.0):
    """
    Display the optimized squad and weekly lineups with bench information
    """
    if squad is None:
        return

    # Determine number of fixtures from the data if not provided
    if num_fixtures is None:
        num_fixtures = len(weekly_lineups)

    fixtures = list(weekly_lineups.keys())

    print("\n" + "=" * 120)
    print(f"OPTIMIZED FPL SQUAD FOR {num_fixtures} FIXTURE(S): {', '.join(fixtures)}")
    print(f"FIXTURE WEIGHTS: {[f'{w:.2f}' for w in weights]}")
    print("=" * 120)

    # Display weekly lineups with bench
    print("\n" + "=" * 120)
    print(f"WEEKLY STARTING LINEUPS & BENCH ({num_fixtures} fixture(s))")
    print("=" * 120)

    for i, fixture in enumerate(fixtures):
        lineup = weekly_lineups[fixture]
        bench = weekly_benches.get(fixture, pd.DataFrame())
        captain = weekly_captains.get(fixture)
        fixture_col = f'{fixture} XP'
        weight = weights[i]

        print(f"\n{fixture} STARTING XI (Weight: {weight:.2f}):")
        print("-" * 80)

        # Sort lineup by position
        lineup_sorted = lineup.sort_values(['Position', fixture_col], ascending=[True, False])

        fixture_total = 0
        captain_bonus = 0

        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = lineup_sorted[lineup_sorted['Position'] == pos]
            if len(pos_players) > 0:
                print(f"\n{pos} ({len(pos_players)}):")
                for _, player in pos_players.iterrows():
                    is_captain = captain is not None and player.name == captain.name
                    captain_mark = " (C)" if is_captain else ""
                    player_points = player[fixture_col]
                    weighted_points = player_points * weight

                    print(f"  {(player['Player Name'] + captain_mark):<30} "
                          f"{player_points:.2f} pts (weighted: {weighted_points:.2f}) "
                          f"{player['Team']}")
                    fixture_total += player_points
                    if is_captain:
                        captain_bonus += player_points

        # Display bench
        if not bench.empty:
            print(f"\nBENCH ({len(bench)}):")
            bench_sorted = bench.sort_values(['Position', fixture_col], ascending=[True, False])
            bench_total = 0
            for _, player in bench_sorted.iterrows():
                player_points = player[fixture_col]
                weighted_points = player_points * weight
                bench_total += player_points
                print(f"  {player['Player Name']:<30} "
                      f"{player_points:.2f} pts (weighted: {weighted_points:.2f}) "
                      f"{player['Team']}")

        fixture_total_with_captain = fixture_total + captain_bonus
        weighted_fixture_total = fixture_total_with_captain * weight
        weighted_captain_bonus = captain_bonus * weight

        print(f"\n{fixture} Starting Total: {fixture_total_with_captain:.2f} pts "
              f"(weighted: {weighted_fixture_total:.2f})")
        print(f"  Base: {fixture_total:.2f} pts (weighted: {fixture_total * weight:.2f})")
        print(f"  Captain: {captain_bonus:.2f} pts (weighted: {weighted_captain_bonus:.2f})")

        if not bench.empty:
            weighted_bench_total = bench_total * weight
            print(f"  Bench: {bench_total:.2f} pts (weighted: {weighted_bench_total:.2f})")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    base_weighted = total_weighted_points - total_captain_bonus_weighted
    base_unweighted = total_unweighted_points - total_captain_bonus_unweighted

    print(f"WEIGHTED SCORING:")
    print(f"  Total Points ({num_fixtures} gameweeks): {total_weighted_points:.2f}")
    print(f"  Base Points: {base_weighted:.2f}")
    print(f"  Captain Bonuses: {total_captain_bonus_weighted:.2f}")
    print(f"  Bench Value: {total_bench_value_weighted:.2f} (not counted in main total)")
    print(f"  Average Points per Gameweek: {total_weighted_points / num_fixtures:.2f}")

    print(f"\nUNWEIGHTED SCORING (for comparison):")
    print(f"  Total Points ({num_fixtures} gameweeks): {total_unweighted_points:.2f}")
    print(f"  Base Points: {base_unweighted:.2f}")
    print(f"  Captain Bonuses: {total_captain_bonus_unweighted:.2f}")
    print(f"  Bench Value: {total_bench_value_unweighted:.2f} (not counted in main total)")
    print(f"  Average Points per Gameweek: {total_unweighted_points / num_fixtures:.2f}")

    # Captain analysis
    print(f"\nCAPTAIN ROTATION:")
    captain_usage = {}
    for i, fixture in enumerate(fixtures):
        if fixture in weekly_captains:
            captain = weekly_captains[fixture]
            captain_name = captain['Player Name']
            fixture_col = f'{fixture} XP'
            captain_points = captain[fixture_col]
            weight = weights[i]
            weighted_captain_points = captain_points * weight

            if captain_name not in captain_usage:
                captain_usage[captain_name] = []
            captain_usage[captain_name].append((fixture, captain_points, weighted_captain_points))

            print(f"  {fixture}: {captain_name} ({captain_points:.2f} pts, "
                  f"weighted: {weighted_captain_points:.2f})")

    # NEW: Bench analysis
    print(f"\nBENCH ANALYSIS:")
    all_bench_players = set()
    bench_appearances = {}

    for fixture in fixtures:
        if fixture in weekly_benches:
            bench = weekly_benches[fixture]
            for _, player in bench.iterrows():
                player_name = player['Player Name']
                all_bench_players.add(player_name)
                if player_name not in bench_appearances:
                    bench_appearances[player_name] = 0
                bench_appearances[player_name] += 1

    if bench_appearances:
        print(f"  Players who appeared on bench:")
        for player_name, appearances in sorted(bench_appearances.items(), key=lambda x: x[1], reverse=True):
            print(f"    {player_name}: {appearances}/{num_fixtures} gameweeks")

    avg_bench_value_weighted = total_bench_value_weighted / num_fixtures if num_fixtures > 0 else 0
    avg_bench_value_unweighted = total_bench_value_unweighted / num_fixtures if num_fixtures > 0 else 0
    print(
        f"  Average bench value per gameweek: {avg_bench_value_unweighted:.2f} pts (weighted: {avg_bench_value_weighted:.2f})")

    print("=" * 120)

    # Display full squad
    print("\nFULL SQUAD (15 players):")
    print("-" * 80)
    squad_sorted = squad.sort_values(['Position', 'Cost'], ascending=[True, False])

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = squad_sorted[squad_sorted['Position'] == pos]
        print(f"\n{pos}:")
        for _, player in pos_players.iterrows():
            # Show both weighted and unweighted totals
            weighted_total = player['Weighted_Total_XP']
            unweighted_total = player['Base_Total_XP']
            print(f"  {player['Player Name']:<25} {player['Cost']:.1f}m "
                  f"Weighted: {weighted_total:.2f} pts  Unweighted: {unweighted_total:.2f} pts "
                  f"{player['Team']}")

    print(f"\nSquad Cost: £{squad_cost:.1f}m (Budget: £{total_squad_cost}m)")
    print(f"Remaining Budget: £{total_squad_cost - squad_cost:.1f}m")


def find_multiple_optimal_teams_weighted_with_bench(excel_file, num_fixtures=5, num_teams=10, sheet_name='Players',
                                                    diversity_method='starting_players', points_tolerance=1.0,
                                                    fixture_weights=None, bench_weight=0.1, total_squad_cost=100.0):
    """
    Find multiple near-optimal teams using different diversity strategies with fixture weighting and bench value

    Parameters:
    - excel_file: Path to Excel file with player data
    - num_fixtures: Number of fixtures to optimize for (1-5)
    - num_teams: Number of different optimal teams to find
    - sheet_name: Excel sheet name containing player data
    - diversity_method: 'starting_players', 'squad_players', or 'points_threshold'
    - points_tolerance: How many points below optimal to allow (default=1.0)
    - fixture_weights: List of weights for each fixture (default: [1.0, 0.85, 0.7, 0.55, 0.4])
    - bench_weight: Weight for bench players' total XP value (default: 0.1)

    Returns:
    - List of teams with their details
    """

    # Set default weights if not provided
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    print(f"Finding top {num_teams} teams for {num_fixtures} fixture(s) using {diversity_method} method...")
    print(f"Points tolerance: {points_tolerance} points below optimal")
    print(f"Using fixture weights: {fixture_weights[:num_fixtures]}")
    print(f"Using bench weight: {bench_weight}")

    # Load data first to get player info
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = df[df['Total XP'] > 0]
    df.columns = df.columns.str.strip()

    all_teams = []
    excluded_starting_lineups = []  # Store starting lineups to force diversity
    excluded_squads = []  # Store squads to exclude

    # Find the optimal points first
    print("Finding optimal points threshold...")
    result = optimise_fpl_team_with_weekly_lineups_and_bench_value(excel_file, num_fixtures, sheet_name,
                                                                   fixture_weights, bench_weight)
    if result[0] is None:
        print("Could not find initial optimal team!")
        return []

    optimal_weighted_points = result[5]  # total_weighted_points (updated index)
    print(f"Optimal weighted points: {optimal_weighted_points:.2f}")

    # Set tolerance for "near-optimal" teams
    min_acceptable_points = optimal_weighted_points - points_tolerance
    print(f"Minimum acceptable weighted points: {min_acceptable_points:.2f}")

    for team_rank in range(1, num_teams + 1):
        print(f"\nFinding team #{team_rank}...")

        if team_rank == 1:
            # For the first team, just use the standard optimization
            result = optimise_fpl_team_with_weekly_lineups_and_bench_value(
                excel_file, num_fixtures, sheet_name, fixture_weights, bench_weight, total_squad_cost=total_squad_cost
            )
        else:

            if diversity_method == 'starting_players':
                # Force different starting lineups
                result = optimise_with_starting_exclusions_weighted_with_bench(excel_file, num_fixtures,
                                                                               excluded_starting_lineups,
                                                                               min_acceptable_points, sheet_name,
                                                                               fixture_weights, bench_weight)
            elif diversity_method == 'squad_players':
                # Force different squads
                result = optimise_with_squad_exclusions_weighted_with_bench(excel_file, num_fixtures, excluded_squads,
                                                                            min_acceptable_points, sheet_name,
                                                                            fixture_weights, bench_weight)
            else:  # points_threshold
                # Allow slightly suboptimal teams for more diversity
                result = optimise_with_relaxed_constraints_weighted_with_bench(excel_file, num_fixtures,
                                                                               excluded_starting_lineups,
                                                                               min_acceptable_points, sheet_name,
                                                                               fixture_weights, bench_weight)

        if result[0] is None:
            print(f"Could not find team #{team_rank} with {min_acceptable_points:.2f}+ points.")
            if points_tolerance < 3.0:  # Try relaxing further
                print(f"Trying with relaxed points tolerance...")
                relaxed_min_points = optimal_weighted_points - (points_tolerance + 1.0)
                if diversity_method == 'starting_players':
                    result = optimise_with_starting_exclusions_weighted_with_bench(excel_file, num_fixtures,
                                                                                   excluded_starting_lineups,
                                                                                   relaxed_min_points, sheet_name,
                                                                                   fixture_weights, bench_weight,
                                                                                   total_squad_cost)
                elif diversity_method == 'squad_players':
                    result = optimise_with_squad_exclusions_weighted_with_bench(excel_file, num_fixtures,
                                                                                excluded_squads,
                                                                                relaxed_min_points, sheet_name,
                                                                                fixture_weights, bench_weight)
                else:
                    result = optimise_with_relaxed_constraints_weighted_with_bench(excel_file, num_fixtures,
                                                                                   excluded_starting_lineups,
                                                                                   relaxed_min_points, sheet_name,
                                                                                   fixture_weights, bench_weight,
                                                                                   total_squad_cost)

                if result[0] is not None:
                    print(f"Found team #{team_rank} with relaxed constraint ({relaxed_min_points:.2f}+ points)")
                else:
                    print(f"Still could not find team #{team_rank}. Stopping search.")
                    break
            else:
                print("Stopping search.")
                break

        (squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
         total_weighted_points, total_captain_bonus_weighted,
         total_unweighted_points, total_captain_bonus_unweighted,
         total_bench_value_weighted, total_bench_value_unweighted, weights) = result

        # Check if this team is genuinely different enough
        if diversity_method == 'starting_players':
            # Store the starting lineups to force future teams to be different
            team_starting_signature = set()
            for fixture in weekly_lineups:
                for player_idx in weekly_lineups[fixture].index:
                    team_starting_signature.add((fixture, player_idx))
            excluded_starting_lineups.append(team_starting_signature)

        # Store this team's details
        team_info = {
            'rank': team_rank,
            'squad': squad,
            'weekly_lineups': weekly_lineups,
            'weekly_captains': weekly_captains,
            'weekly_benches': weekly_benches,
            'squad_cost': squad_cost,
            'total_weighted_points': total_weighted_points,
            'total_captain_bonus_weighted': total_captain_bonus_weighted,
            'total_unweighted_points': total_unweighted_points,
            'total_captain_bonus_unweighted': total_captain_bonus_unweighted,
            'total_bench_value_weighted': total_bench_value_weighted,
            'total_bench_value_unweighted': total_bench_value_unweighted,
            'weights': weights,
            'squad_indices': set(squad.index)
        }

        all_teams.append(team_info)
        excluded_squads.append(set(squad.index))

        points_diff = optimal_weighted_points - total_weighted_points
        print(f"Team #{team_rank}: {total_weighted_points:.2f} weighted pts (Δ-{points_diff:.2f}), £{squad_cost:.1f}m")

        # Dynamic tolerance adjustment - if we're finding good alternatives, continue
        if team_rank > 1 and points_diff > points_tolerance * 1.5:
            print(f"Points gap widening significantly. Consider stopping here for quality.")

    return all_teams


def optimise_with_starting_exclusions_weighted_with_bench(excel_file, num_fixtures, excluded_starting_lineups,
                                                          min_points,
                                                          sheet_name='Players', fixture_weights=None,
                                                          bench_weight=0.1, total_squad_cost=100.0):
    """
    Optimize while forcing different starting lineups from previous solutions with fixture weighting and bench value
    """
    # Set default weights if not provided
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    weights = fixture_weights[:num_fixtures]

    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = df[df['Total XP'] > 0]
    df.columns = df.columns.str.strip()

    # Validate num_fixtures parameter
    if not isinstance(num_fixtures, int) or num_fixtures < 1 or num_fixtures > 6:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Define fixture columns
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP']
    fixture_columns = all_fixture_columns[:num_fixtures]
    fixtures = [f'F{i + 1}' for i in range(num_fixtures)]

    # Check if all fixture columns exist
    missing_columns = [col for col in fixture_columns if col not in df.columns]
    if missing_columns:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Calculate weighted and base totals
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

    df['Base_Total_XP'] = df[fixture_columns].sum(axis=1)

    # Create the optimisation problem
    prob = pulp.LpProblem("FPL_Diverse_Starting_Lineups_Weighted_With_Bench", pulp.LpMaximize)

    # Decision variables
    squad_vars = {}
    for i in df.index:
        squad_vars[i] = pulp.LpVariable(f"squad_{i}", cat='Binary')

    starting_vars = {}
    for fixture in fixtures:
        starting_vars[fixture] = {}
        for i in df.index:
            starting_vars[fixture][i] = pulp.LpVariable(f"starting_{fixture}_{i}", cat='Binary')

    captain_vars = {}
    for fixture in fixtures:
        captain_vars[fixture] = {}
        for i in df.index:
            captain_vars[fixture][i] = pulp.LpVariable(f"captain_{fixture}_{i}", cat='Binary')

    # NEW: Bench variables
    bench_vars = {}
    for fixture in fixtures:
        bench_vars[fixture] = {}
        for i in df.index:
            bench_vars[fixture][i] = pulp.LpVariable(f"bench_{fixture}_{i}", cat='Binary')

    # Objective function with weighting and bench value
    objective_terms = []
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        for player_idx in df.index:
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # NEW: Bench value
            objective_terms.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

    prob += pulp.lpSum(objective_terms)

    # All standard constraints (same as before)
    prob += pulp.lpSum([squad_vars[i] for i in df.index]) == 15
    prob += pulp.lpSum([df.loc[i, 'Cost'] * squad_vars[i] for i in df.index]) <= total_squad_cost

    squad_position_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for position, required_count in squad_position_requirements.items():
        position_players = df[df['Position'] == position].index
        prob += pulp.lpSum([squad_vars[i] for i in position_players]) == required_count

    unique_teams = df['Team'].unique()
    for team in unique_teams:
        team_players = df[df['Team'] == team].index
        prob += pulp.lpSum([squad_vars[i] for i in team_players]) <= 3

    # budget_gks = df[(df['Position'] == 'GK') & (df['Cost'] <= 4.0)].index
    # if len(budget_gks) > 0:
    #     prob += pulp.lpSum([squad_vars[i] for i in budget_gks]) >= 1

    for fixture in fixtures:
        prob += pulp.lpSum([starting_vars[fixture][i] for i in df.index]) == 11

        for i in df.index:
            prob += starting_vars[fixture][i] <= squad_vars[i]

        gk_players = df[df['Position'] == 'GK'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in gk_players]) == 1

        def_players = df[df['Position'] == 'DEF'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) <= 5

        mid_players = df[df['Position'] == 'MID'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) <= 5

        fwd_players = df[df['Position'] == 'FWD'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) >= 1
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) <= 3

    for fixture in fixtures:
        prob += pulp.lpSum([captain_vars[fixture][i] for i in df.index]) == 1
        for i in df.index:
            prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

    # NEW: Bench constraints
    for fixture in fixtures:
        for i in df.index:
            prob += bench_vars[fixture][i] <= squad_vars[i]
            prob += bench_vars[fixture][i] <= 1 - starting_vars[fixture][i]
            prob += bench_vars[fixture][i] >= squad_vars[i] - starting_vars[fixture][i]

    # NEW: Force different starting lineups
    for excluded_lineup in excluded_starting_lineups:
        # Count how many of the excluded starting decisions we're making
        excluded_starting_vars = []
        for fixture, player_idx in excluded_lineup:
            if fixture in starting_vars and player_idx in starting_vars[fixture]:
                excluded_starting_vars.append(starting_vars[fixture][player_idx])

        # Ensure we don't replicate the exact same starting lineup
        if excluded_starting_vars:
            prob += pulp.lpSum(excluded_starting_vars) <= len(excluded_starting_vars) - 1

    # Add minimum weighted points constraint
    total_weighted_points_expr = []
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        for player_idx in df.index:
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # NEW: Include bench value in minimum points constraint
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

    prob += pulp.lpSum(total_weighted_points_expr) >= min_points

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.LpStatusOptimal:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Extract results
    squad_indices = []
    for i in df.index:
        if squad_vars[i].varValue == 1:
            squad_indices.append(i)
    squad = df.loc[squad_indices].copy()

    weekly_lineups = {}
    for fixture in fixtures:
        starting_indices = []
        for i in df.index:
            if starting_vars[fixture][i].varValue == 1:
                starting_indices.append(i)
        weekly_lineups[fixture] = df.loc[starting_indices].copy()

    weekly_captains = {}
    for fixture in fixtures:
        for i in df.index:
            if captain_vars[fixture][i].varValue == 1:
                weekly_captains[fixture] = df.loc[i]
                break

    # NEW: Extract bench results
    weekly_benches = {}
    for fixture in fixtures:
        bench_indices = []
        for i in df.index:
            if bench_vars[fixture][i].varValue == 1:
                bench_indices.append(i)
        weekly_benches[fixture] = df.loc[bench_indices].copy()

    squad_cost = squad['Cost'].sum()
    total_weighted_points = 0
    total_unweighted_points = 0
    total_captain_bonus_weighted = 0
    total_captain_bonus_unweighted = 0
    total_bench_value_weighted = 0
    total_bench_value_unweighted = 0

    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]

        # Points from starters
        fixture_points = weekly_lineups[fixture][fixture_col].sum()
        total_unweighted_points += fixture_points
        total_weighted_points += fixture_points * weight

        # Captain bonus
        if fixture in weekly_captains:
            captain_bonus = weekly_captains[fixture][fixture_col]
            total_unweighted_points += captain_bonus
            total_captain_bonus_unweighted += captain_bonus

            weighted_captain_bonus = captain_bonus * weight
            total_weighted_points += weighted_captain_bonus
            total_captain_bonus_weighted += weighted_captain_bonus

        # NEW: Bench value
        if fixture in weekly_benches:
            bench_value = weekly_benches[fixture][fixture_col].sum()
            total_bench_value_unweighted += bench_value
            total_bench_value_weighted += bench_value * weight

    return (squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
            total_weighted_points, total_captain_bonus_weighted,
            total_unweighted_points, total_captain_bonus_unweighted,
            total_bench_value_weighted, total_bench_value_unweighted, weights)


def optimise_with_squad_exclusions_weighted_with_bench(excel_file, num_fixtures, excluded_squads, min_points,
                                                       sheet_name='Players', fixture_weights=None, bench_weight=0.1,
                                                       total_squad_cost=100.0):
    """
    Optimize with squad exclusions, fixture weighting, and bench value
    """
    # Set default weights if not provided
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    weights = fixture_weights[:num_fixtures]

    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = df[df['Total XP'] > 0]
    df.columns = df.columns.str.strip()

    # Validate num_fixtures parameter
    if not isinstance(num_fixtures, int) or num_fixtures < 1 or num_fixtures > 6:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Define fixture columns
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP']
    fixture_columns = all_fixture_columns[:num_fixtures]
    fixtures = [f'F{i + 1}' for i in range(num_fixtures)]

    # Check if all fixture columns exist
    missing_columns = [col for col in fixture_columns if col not in df.columns]
    if missing_columns:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Calculate weighted and base totals
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

    df['Base_Total_XP'] = df[fixture_columns].sum(axis=1)

    # Create the optimisation problem
    prob = pulp.LpProblem("FPL_Squad_Exclusions_Weighted_With_Bench", pulp.LpMaximize)

    # Decision variables
    squad_vars = {}
    for i in df.index:
        squad_vars[i] = pulp.LpVariable(f"squad_{i}", cat='Binary')

    starting_vars = {}
    for fixture in fixtures:
        starting_vars[fixture] = {}
        for i in df.index:
            starting_vars[fixture][i] = pulp.LpVariable(f"starting_{fixture}_{i}", cat='Binary')

    captain_vars = {}
    for fixture in fixtures:
        captain_vars[fixture] = {}
        for i in df.index:
            captain_vars[fixture][i] = pulp.LpVariable(f"captain_{fixture}_{i}", cat='Binary')

    # Bench variables
    bench_vars = {}
    for fixture in fixtures:
        bench_vars[fixture] = {}
        for i in df.index:
            bench_vars[fixture][i] = pulp.LpVariable(f"bench_{fixture}_{i}", cat='Binary')

    # Objective function with weighting and bench value
    objective_terms = []
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        for player_idx in df.index:
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # Bench value
            objective_terms.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

    prob += pulp.lpSum(objective_terms)

    # All standard constraints
    prob += pulp.lpSum([squad_vars[i] for i in df.index]) == 15
    prob += pulp.lpSum([df.loc[i, 'Cost'] * squad_vars[i] for i in df.index]) <= total_squad_cost

    squad_position_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for position, required_count in squad_position_requirements.items():
        position_players = df[df['Position'] == position].index
        prob += pulp.lpSum([squad_vars[i] for i in position_players]) == required_count

    unique_teams = df['Team'].unique()
    for team in unique_teams:
        team_players = df[df['Team'] == team].index
        prob += pulp.lpSum([squad_vars[i] for i in team_players]) <= 3

    # Weekly lineup constraints
    for fixture in fixtures:
        prob += pulp.lpSum([starting_vars[fixture][i] for i in df.index]) == 11

        for i in df.index:
            prob += starting_vars[fixture][i] <= squad_vars[i]

        gk_players = df[df['Position'] == 'GK'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in gk_players]) == 1

        def_players = df[df['Position'] == 'DEF'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in def_players]) <= 5

        mid_players = df[df['Position'] == 'MID'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) >= 3
        prob += pulp.lpSum([starting_vars[fixture][i] for i in mid_players]) <= 5

        fwd_players = df[df['Position'] == 'FWD'].index
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) >= 1
        prob += pulp.lpSum([starting_vars[fixture][i] for i in fwd_players]) <= 3

    # Captain constraints
    for fixture in fixtures:
        prob += pulp.lpSum([captain_vars[fixture][i] for i in df.index]) == 1
        for i in df.index:
            prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

    # Bench constraints
    for fixture in fixtures:
        for i in df.index:
            prob += bench_vars[fixture][i] <= squad_vars[i]
            prob += bench_vars[fixture][i] <= 1 - starting_vars[fixture][i]
            prob += bench_vars[fixture][i] >= squad_vars[i] - starting_vars[fixture][i]

    # Squad exclusion constraints
    for excluded_squad in excluded_squads:
        # Ensure we don't select the exact same squad
        excluded_squad_vars = []
        for player_idx in excluded_squad:
            if player_idx in squad_vars:
                excluded_squad_vars.append(squad_vars[player_idx])

        # Force at least one different player in the squad
        if excluded_squad_vars:
            prob += pulp.lpSum(excluded_squad_vars) <= len(excluded_squad_vars) - 1

    # Add minimum weighted points constraint
    total_weighted_points_expr = []
    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]
        for player_idx in df.index:
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # Include bench value in minimum points constraint
            total_weighted_points_expr.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

    prob += pulp.lpSum(total_weighted_points_expr) >= min_points

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.LpStatusOptimal:
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Extract results
    squad_indices = []
    for i in df.index:
        if squad_vars[i].varValue == 1:
            squad_indices.append(i)
    squad = df.loc[squad_indices].copy()

    weekly_lineups = {}
    for fixture in fixtures:
        starting_indices = []
        for i in df.index:
            if starting_vars[fixture][i].varValue == 1:
                starting_indices.append(i)
        weekly_lineups[fixture] = df.loc[starting_indices].copy()

    weekly_captains = {}
    for fixture in fixtures:
        for i in df.index:
            if captain_vars[fixture][i].varValue == 1:
                weekly_captains[fixture] = df.loc[i]
                break

    # Extract bench results
    weekly_benches = {}
    for fixture in fixtures:
        bench_indices = []
        for i in df.index:
            if bench_vars[fixture][i].varValue == 1:
                bench_indices.append(i)
        weekly_benches[fixture] = df.loc[bench_indices].copy()

    squad_cost = squad['Cost'].sum()
    total_weighted_points = 0
    total_unweighted_points = 0
    total_captain_bonus_weighted = 0
    total_captain_bonus_unweighted = 0
    total_bench_value_weighted = 0
    total_bench_value_unweighted = 0

    for i, fixture in enumerate(fixtures):
        fixture_col = f'{fixture} XP'
        weight = weights[i]

        # Points from starters
        fixture_points = weekly_lineups[fixture][fixture_col].sum()
        total_unweighted_points += fixture_points
        total_weighted_points += fixture_points * weight

        # Captain bonus
        if fixture in weekly_captains:
            captain_bonus = weekly_captains[fixture][fixture_col]
            total_unweighted_points += captain_bonus
            total_captain_bonus_unweighted += captain_bonus

            weighted_captain_bonus = captain_bonus * weight
            total_weighted_points += weighted_captain_bonus
            total_captain_bonus_weighted += weighted_captain_bonus

        # Bench value
        if fixture in weekly_benches:
            bench_value = weekly_benches[fixture][fixture_col].sum()
            total_bench_value_unweighted += bench_value
            total_bench_value_weighted += bench_value * weight

    return (squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
            total_weighted_points, total_captain_bonus_weighted,
            total_unweighted_points, total_captain_bonus_unweighted,
            total_bench_value_weighted, total_bench_value_unweighted, weights)


def optimise_with_relaxed_constraints_weighted_with_bench(excel_file, num_fixtures, excluded_starting_lineups,
                                                          min_points,
                                                          sheet_name='Players', fixture_weights=None, bench_weight=0.1,
                                                          total_squad_cost=100.0):
    """
    Allow slightly suboptimal teams for more diversity with fixture weighting and bench value
    """
    return optimise_with_starting_exclusions_weighted_with_bench(excel_file, num_fixtures, excluded_starting_lineups,
                                                                 min_points, sheet_name, fixture_weights, bench_weight,
                                                                 total_squad_cost)


def analyse_top_teams_weighted_with_bench(all_teams, num_fixtures):
    """
    Analyse which players appear most frequently in top teams (weighted version with bench info)
    """
    if not all_teams:
        return

    print(f"\n" + "=" * 100)
    print(f"ANALYSIS OF TOP {len(all_teams)} TEAMS (WEIGHTED WITH BENCH VALUE)")
    print("=" * 100)

    # Show ranking of all teams
    print(f"\nTEAM RANKINGS:")
    print(f"{'Rank':<6} {'Weighted Pts':<12} {'Unweighted Pts':<14} {'Bench Value':<12} {'Cost':<8} {'Avg/GW (W)':<12}")
    print("-" * 85)

    for team in all_teams:
        avg_per_gw_weighted = team['total_weighted_points'] / num_fixtures
        avg_bench_value = team['total_bench_value_weighted'] / num_fixtures
        print(f"{team['rank']:<6} {team['total_weighted_points']:<12.2f} "
              f"{team['total_unweighted_points']:<14.2f} {avg_bench_value:<12.2f} £{team['squad_cost']:<7.1f}m "
              f"{avg_per_gw_weighted:<12.2f}")

    # Player frequency analysis
    player_appearances = {}
    player_names = {}

    if all_teams:
        for team in all_teams:
            for _, player in team['squad'].iterrows():
                player_name = player['Player Name']
                player_id = player.name
                player_names[player_id] = player_name

                if player_id not in player_appearances:
                    player_appearances[player_id] = 0
                player_appearances[player_id] += 1

    # Sort players by frequency
    sorted_players = sorted(player_appearances.items(), key=lambda x: x[1], reverse=True)

    print(f"\nMOST SELECTED PLAYERS (appearing in top {len(all_teams)} teams):")
    print(f"{'Player':<30} {'Appearances':<12} {'Frequency':<10}")
    print("-" * 55)

    for player_id, count in sorted_players[:20]:  # Show top 20 most frequent
        player_name = player_names.get(player_id, f"Player {player_id}")
        frequency = (count / len(all_teams)) * 100
        print(f"{player_name:<30} {count}/{len(all_teams):<11} {frequency:<10.1f}%")

    # Essential players (appear in 80%+ of teams)
    essential_threshold = len(all_teams) * 0.8
    essential_players = [(player_id, count) for player_id, count in sorted_players if count >= essential_threshold]

    if essential_players:
        print(f"\nESSENTIAL PLAYERS (in {essential_threshold / len(all_teams) * 100:.0f}%+ of teams):")
        for player_id, count in essential_players:
            player_name = player_names.get(player_id, f"Player {player_id}")
            print(f"  {player_name} ({count}/{len(all_teams)} teams)")

    # Position analysis
    position_frequency = {}
    for team in all_teams:
        for _, player in team['squad'].iterrows():
            position = player['Position']
            player_name = player['Player Name']

            if position not in position_frequency:
                position_frequency[position] = {}

            if player_name not in position_frequency[position]:
                position_frequency[position][player_name] = 0
            position_frequency[position][player_name] += 1

    print(f"\nTOP PLAYERS BY POSITION:")
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        if position in position_frequency:
            print(f"\n{position}:")
            sorted_pos_players = sorted(position_frequency[position].items(), key=lambda x: x[1], reverse=True)
            for player_name, count in sorted_pos_players[:5]:  # Top 5 per position
                frequency = (count / len(all_teams)) * 100
                print(f"  {player_name:<25} {count}/{len(all_teams)} ({frequency:.1f}%)")

    # NEW: Bench analysis
    print(f"\nBENCH APPEARANCE ANALYSIS:")
    bench_appearances = {}
    starting_appearances = {}

    for team in all_teams:
        # Track who appears on bench
        for fixture in team['weekly_benches']:
            bench = team['weekly_benches'][fixture]
            for _, player in bench.iterrows():
                player_name = player['Player Name']
                if player_name not in bench_appearances:
                    bench_appearances[player_name] = 0
                bench_appearances[player_name] += 1

        # Track who starts
        for fixture in team['weekly_lineups']:
            lineup = team['weekly_lineups'][fixture]
            for _, player in lineup.iterrows():
                player_name = player['Player Name']
                if player_name not in starting_appearances:
                    starting_appearances[player_name] = 0
                starting_appearances[player_name] += 1

    # Players who frequently appear on bench
    if bench_appearances:
        sorted_bench = sorted(bench_appearances.items(), key=lambda x: x[1], reverse=True)
        print(f"\nMOST FREQUENT BENCH PLAYERS:")
        print(f"{'Player':<30} {'Bench Apps':<12} {'Start Apps':<12} {'Bench %':<10}")
        print("-" * 70)

        for player_name, bench_count in sorted_bench[:10]:
            start_count = starting_appearances.get(player_name, 0)
            total_apps = bench_count + start_count
            bench_percentage = (bench_count / total_apps * 100) if total_apps > 0 else 0
            print(f"{player_name:<30} {bench_count:<12} {start_count:<12} {bench_percentage:<10.1f}%")


def display_specific_team_weighted_with_bench(team_info, num_fixtures):
    """
    Display details for a specific team from the multiple teams analysis (weighted version with bench)
    """
    print(f"\n" + "=" * 100)
    print(f"TEAM #{team_info['rank']} DETAILS - {team_info['total_weighted_points']:.2f} WEIGHTED POINTS")
    print("=" * 100)

    display_weighted_lineup_results_with_bench(
        team_info['squad'],
        team_info['weekly_lineups'],
        team_info['weekly_captains'],
        team_info['weekly_benches'],
        team_info['squad_cost'],
        team_info['total_weighted_points'],
        team_info['total_captain_bonus_weighted'],
        team_info['total_unweighted_points'],
        team_info['total_captain_bonus_unweighted'],
        team_info['total_bench_value_weighted'],
        team_info['total_bench_value_unweighted'],
        team_info['weights'],
        num_fixtures
    )


def main_weighted_with_bench_value(num_fixtures=5, fixture_weights=None, bench_weight=0.1,
                                   find_multiple=False, num_teams=10, diversity_method='starting_players',
                                   points_tolerance=1.0, total_squad_cost=100.0):

    excel_file = "Fantasy Premier League.xlsx"

    # Set default weights if not provided
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    print(f"Starting WEIGHTED FPL optimisation with BENCH VALUE for {num_fixtures} fixture(s)...")
    print(f"Using fixture weights: {fixture_weights[:num_fixtures]}")
    print(f"Using bench weight: {bench_weight}")

    try:
        if find_multiple:
            # Find multiple teams and analyze
            all_teams = find_multiple_optimal_teams_weighted_with_bench(excel_file, num_fixtures, num_teams,
                                                                        diversity_method=diversity_method,
                                                                        points_tolerance=points_tolerance,
                                                                        fixture_weights=fixture_weights,
                                                                        bench_weight=bench_weight,
                                                                        total_squad_cost=total_squad_cost)

            if all_teams:
                # Show analysis of all teams
                analyse_top_teams_weighted_with_bench(all_teams, num_fixtures)

                # Show detailed view of top 3 teams
                num_detailed_teams = min(3, len(all_teams))
                print(f"\nDETAILED VIEW OF TOP {num_detailed_teams} TEAMS:")
                print("=" * 120)

                for i in range(num_detailed_teams):
                    if i > 0:  # Add separator between teams
                        print("\n" + "=" * 120)
                    display_specific_team_weighted_with_bench(all_teams[i], num_fixtures)

                return all_teams
            else:
                print("No teams found!")
                return None
        else:
            # Single team optimisation with weighting and bench value
            result = optimise_fpl_team_with_weekly_lineups_and_bench_value(
                excel_file, num_fixtures, fixture_weights=fixture_weights, bench_weight=bench_weight,
                total_squad_cost=total_squad_cost
            )

            if result[0] is not None:
                (squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
                 total_weighted_points, total_captain_bonus_weighted,
                 total_unweighted_points, total_captain_bonus_unweighted,
                 total_bench_value_weighted, total_bench_value_unweighted, weights) = result

                # Display results
                display_weighted_lineup_results_with_bench(
                    squad, weekly_lineups, weekly_captains, weekly_benches, squad_cost,
                    total_weighted_points, total_captain_bonus_weighted,
                    total_unweighted_points, total_captain_bonus_unweighted,
                    total_bench_value_weighted, total_bench_value_unweighted,
                    weights, num_fixtures, total_squad_cost
                )

                return squad, weekly_lineups, weekly_captains, weekly_benches, weights
            else:
                print("No optimal solution found!")
                return None

    except FileNotFoundError:
        print(f"Error: Could not find {excel_file}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    fixture_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print("=== SINGLE TEAM OPTIMISATION ===")
    result = main_weighted_with_bench_value(num_fixtures=6, fixture_weights=fixture_weights, bench_weight=0.05,
                                            total_squad_cost=100)

    # print("\n\n=== MULTIPLE TEAMS OPTIMISATION ===")
    # all_teams = main_weighted_with_bench_value(num_fixtures=6, fixture_weights=fixture_weights,
    #                                            bench_weight=0.2, find_multiple=True,
    #                                            num_teams=15, diversity_method='starting_players',
    #                                            points_tolerance=10.0)
