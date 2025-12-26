import pandas as pd
import pulp


def load_current_team(excel_file, sheet_name='GW Teams'):
    print("Loading current team from GW Teams sheet...")

    # Read the GW Teams sheet
    df_teams = pd.read_excel(excel_file, sheet_name=sheet_name)

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
    df = pd.read_excel(excel_file, sheet_name=players_sheet)
    df.columns = df.columns.str.strip()

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
    df = pd.read_excel(excel_file, sheet_name=players_sheet)
    df.columns = df.columns.str.strip()

    # Define fixture columns
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP']
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


def calculate_optimised_baseline(df, current_team_indices, fixtures, weights, bench_weights, gk_bench_weights):
    """
    Calculate the optimal baseline score for the current squad by running a mini-optimizer.
    This simulates dynamic lineup selection across fixtures without any transfers.

    Parameters:
    - bench_weights: array of bench weights for each fixture (e.g., [0.3, 0.3, 0.25, 0.2, 0.15, 0.1])
    - gk_bench_weights: array of GK bench weights for each fixture

    Returns: (total_weighted_points, total_starting_xi_points, f1_squad_total, f1_starting_xi_total)
    """
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
        bench_weight = bench_weights[i]
        gk_bench_weight = gk_bench_weights[i]

        for player_idx in current_team_indices:
            # Starting points
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
            # Captain bonus
            objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])
            # Bench value
            is_gk = df.loc[player_idx, 'Position'] == 'GK'
            current_bench_weight = gk_bench_weight if is_gk else bench_weight
            objective_terms.append(
                df.loc[player_idx, fixture_col] * weight * current_bench_weight * bench_vars[fixture][player_idx])

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
        bench_weight = bench_weights[i]
        gk_bench_weight = gk_bench_weights[i]

        for player_idx in current_team_indices:
            if starting_vars[fixture][player_idx].varValue == 1:
                total_starting_points += df.loc[player_idx, fixture_col] * weight
                if i == 0:  # F1
                    f1_starting += df.loc[player_idx, fixture_col]

            if captain_vars[fixture][player_idx].varValue == 1:
                total_captain_points += df.loc[player_idx, fixture_col] * weight
                if i == 0:  # F1
                    f1_captain += df.loc[player_idx, fixture_col]

            if bench_vars[fixture][player_idx].varValue == 1:
                is_gk = df.loc[player_idx, 'Position'] == 'GK'
                current_bench_weight = gk_bench_weight if is_gk else bench_weight
                total_bench_points += df.loc[player_idx, fixture_col] * weight * current_bench_weight
                if i == 0:  # F1
                    f1_bench += df.loc[player_idx, fixture_col] * current_bench_weight

    total_squad_weighted = total_starting_points + total_captain_points + total_bench_points
    total_starting_xi_weighted = total_starting_points + total_captain_points

    f1_squad_total = f1_starting + f1_captain + f1_bench
    f1_starting_xi_total = f1_starting + f1_captain

    return total_squad_weighted, total_starting_xi_weighted, f1_squad_total, f1_starting_xi_total


def optimise_transfers_multi(excel_file, current_team_names, max_transfers=2, num_fixtures=5,
                             fixture_weights=None, players_sheet='Players',
                             additional_budget=0.0, bench_weights=None, gk_bench_weights=None,
                             force_transfer_out=None, num_solutions=3, max_defensive_players_per_team=3):
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    # Set default bench weights (arrays)
    if bench_weights is None:
        bench_weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

    if gk_bench_weights is None:
        gk_bench_weights = [0.10, 0.10, 0.08, 0.06, 0.04, 0.02]

    weights = fixture_weights[:num_fixtures]
    bench_weights = bench_weights[:num_fixtures]
    gk_bench_weights = gk_bench_weights[:num_fixtures]

    # Handle forced transfers
    if force_transfer_out is None:
        force_transfer_out = []

    print(f"\nOptimising transfers: max {max_transfers} transfers for {num_fixtures} fixtures")
    print(f"Finding top {num_solutions} transfer combinations")
    print(f"Using fixture weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional budget available: £{additional_budget:.1f}m")
    print(f"Outfielder bench weights: {[f'{w:.2f}' for w in bench_weights]}")
    print(f"GK bench weights: {[f'{w:.2f}' for w in gk_bench_weights]}")
    print(f"Max defensive players (GK+DEF) per team: {max_defensive_players_per_team}")
    if force_transfer_out:
        print(f"Forced transfers out: {force_transfer_out}")

    # Load player data
    df = pd.read_excel(excel_file, sheet_name=players_sheet)
    df.columns = df.columns.str.strip()

    # Define fixture columns and fixtures
    fixtures = [f'F{i + 1}' for i in range(num_fixtures)]
    fixture_columns = [f'{fixture} XP' for fixture in fixtures]

    # Calculate weighted total XP
    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

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

    # Calculate total budget available
    total_budget = current_team_cost + additional_budget

    # Calculate current team points for comparison
    current_team_df, current_points, _ = analyse_current_team(excel_file, current_team_names,
                                                              num_fixtures, fixture_weights,
                                                              players_sheet, additional_budget)

    # Store all solutions
    all_solutions = []
    excluded_transfer_combinations = []

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
        for i, fixture in enumerate(fixtures):
            fixture_col = f'{fixture} XP'
            weight = weights[i]
            bench_weight = bench_weights[i]
            gk_bench_weight = gk_bench_weights[i]

            for player_idx in df.index:
                # Weighted points from starting players
                objective_terms.append(df.loc[player_idx, fixture_col] * weight * starting_vars[fixture][player_idx])
                # Weighted captain bonus (additional points for captain)
                objective_terms.append(df.loc[player_idx, fixture_col] * weight * captain_vars[fixture][player_idx])

                # Weighted bench value - use different weights for GK vs outfielders
                is_gk = df.loc[player_idx, 'Position'] == 'GK'
                current_bench_weight = gk_bench_weight if is_gk else bench_weight
                objective_terms.append(
                    df.loc[player_idx, fixture_col] * weight * current_bench_weight * bench_vars[fixture][player_idx])

        prob += pulp.lpSum(objective_terms)

        # Constraint 1: Exactly 15 players in squad
        prob += pulp.lpSum([squad_vars[i] for i in df.index]) == 15

        # Constraint 2: Squad position requirements
        squad_position_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for position, required_count in squad_position_requirements.items():
            position_players = df[df['Position'] == position].index
            prob += pulp.lpSum([squad_vars[i] for i in position_players]) == required_count

        # Constraint 3: Maximum 3 players per team
        unique_teams = df['Team'].unique()
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

        for fixture in fixtures:
            starting_lineups[fixture] = []
            bench_players[fixture] = []

            for i in df.index:
                if starting_vars[fixture][i].varValue == 1:
                    starting_lineups[fixture].append(i)
                if captain_vars[fixture][i].varValue == 1:
                    captains[fixture] = i
                if bench_vars[fixture][i].varValue == 1:
                    bench_players[fixture].append(i)

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
            bench_weight = bench_weights[i]
            gk_bench_weight = gk_bench_weights[i]

            # Starting points
            for player_idx in starting_lineups[fixture]:
                total_starting_points += df.loc[player_idx, fixture_col] * weight

            # Captain bonus points
            if fixture in captains:
                captain_idx = captains[fixture]
                total_captain_points += df.loc[captain_idx, fixture_col] * weight

            # Bench points - use different weights for GK vs outfielders
            for player_idx in bench_players[fixture]:
                is_gk = df.loc[player_idx, 'Position'] == 'GK'
                current_bench_weight = gk_bench_weight if is_gk else bench_weight
                total_bench_points += df.loc[player_idx, fixture_col] * weight * current_bench_weight

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

        for player_idx in bench_players[f1_fixture]:
            is_gk = df.loc[player_idx, 'Position'] == 'GK'
            current_bench_weight = gk_bench_weights[0] if is_gk else bench_weights[0]
            f1_bench_points += df.loc[player_idx, 'F1 XP'] * current_bench_weight

        f1_starting_xi_points = f1_starting_points + f1_captain_points
        f1_total_points = f1_starting_xi_points + f1_bench_points

        num_transfers = len(transfers_out)

        # Calculate optimised baseline using mini-optimizer
        # This properly simulates dynamic lineup selection across fixtures
        baseline_result = calculate_optimised_baseline(
            df, current_team_indices, fixtures, weights, bench_weights, gk_bench_weights
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
                bench_weight = bench_weights[fix_idx]
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

                # Bench
                bench_players = [p for p in current_team_xp if p not in starting_xi]
                bench_points = 0
                for p in bench_players:
                    is_gk = p['position'] == 'GK'
                    current_bench_weight = gk_bench_weight if is_gk else bench_weight
                    bench_points += p['xp'] * current_bench_weight

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
            'bench_weights': bench_weights,
            'gk_bench_weights': gk_bench_weights,
            'forced_transfers_out': force_transfer_out,
            'forced_out_indices': forced_out_indices,
            'max_defensive_players_per_team': max_defensive_players_per_team
        }

        all_solutions.append(solution_result)

    return all_solutions


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
                     if data['count'] / total_solutions > 0.5]
    consensus_in = [name for name, data in frequency_analysis['transfers_in']
                    if data['count'] / total_solutions > 0.5]

    if consensus_out:
        print(f"Consensus transfers OUT (>50%): {', '.join(consensus_out)}")
    else:
        print("No consensus transfers out (none appear in >50% of solutions)")

    if consensus_in:
        print(f"Consensus transfers IN (>50%): {', '.join(consensus_in)}")
    else:
        print("No consensus transfers in (none appear in >50% of solutions)")


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
            for _, player in pos_players.iterrows():
                points = player[f'{fixture} XP']
                cost = player['Cost']
                team = player['Team']
                captain_mark = " (C)" if player['Is_Captain'] else ""
                total_starting_points += points
                if player['Is_Captain']:
                    captain_bonus += points

                print(f"  {player['Player Name']:<25} {team:<8} £{cost:.1f}m  {points:.2f} pts{captain_mark}")
            print()

    print("BENCH:")
    bench_sorted = bench_players_data.sort_values(['Position', f'{fixture} XP'], ascending=[True, False])
    total_bench_value = 0

    for _, player in bench_sorted.iterrows():
        points = player[f'{fixture} XP']
        total_bench_value += points
        cost = player['Cost']
        team = player['Team']
        print(f"  {player['Player Name']:<25} {team:<8} £{cost:.1f}m  {points:.2f} pts")

    print(f"\nFIXTURE {fixture_num} SUMMARY:")
    print(f"  Starting XI Points: {total_starting_points + captain_bonus:.2f}")
    print(f"  Bench Points: {total_bench_value:.2f}")


def main_multi_transfer_optimiser(excel_file="Fantasy Premier League.xlsx", max_transfers=2, num_fixtures=5,
                                  fixture_weights=None, show_current_analysis=True,
                                  additional_budget=0.0, bench_weights=None, gk_bench_weights=None,
                                  force_transfer_out=None, num_solutions_display=3,
                                  show_all_details=False, show_detailed_f1=False,
                                  compute_solutions=20, show_frequency_analysis=True,
                                  min_frequency=2, max_defensive_players_per_team=3):
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    if bench_weights is None:
        bench_weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

    if gk_bench_weights is None:
        gk_bench_weights = [0.10, 0.10, 0.08, 0.06, 0.04, 0.02]

    weights = fixture_weights[:num_fixtures]
    bench_weights_used = bench_weights[:num_fixtures]
    gk_bench_weights_used = gk_bench_weights[:num_fixtures]

    print(f"FPL MULTI-TRANSFER OPTIMISER")
    print(f"Max Transfers: {max_transfers}")
    print(f"Fixtures: {num_fixtures}")
    print(f"Computing {compute_solutions} solutions, displaying top {num_solutions_display}")
    print(f"Weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional Budget: £{additional_budget:.1f}m")
    print(f"Outfielder Bench Weights: {[f'{w:.2f}' for w in bench_weights_used]}")
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

        # Load player data for detailed display
        df = pd.read_excel(excel_file, sheet_name='Players')
        df.columns = df.columns.str.strip()

        # Get current team indices
        current_team_indices = []
        for player_name in current_team_names:
            matches = df[df['Player Name'].str.strip() == player_name.strip()]
            if len(matches) == 0:
                matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]
            if len(matches) > 0:
                current_team_indices.append(matches.index[0])

        # Optimise transfers - get multiple solutions
        # Compute more solutions than we display for better frequency analysis
        all_solutions = optimise_transfers_multi(
            excel_file, current_team_names, max_transfers, num_fixtures,
            fixture_weights, 'Players', additional_budget, bench_weights_used,
            gk_bench_weights_used, force_transfer_out, compute_solutions, max_defensive_players_per_team
        )

        if not all_solutions:
            print("No solutions found!")
            return None

        # Show frequency analysis if requested
        if show_frequency_analysis and len(all_solutions) > 1:
            fixtures = all_solutions[0]['fixtures']
            weights = all_solutions[0]['weights']
            frequency_analysis = analyse_transfer_frequency(all_solutions, fixtures, weights)
            display_transfer_frequency(frequency_analysis, min_frequency)

        # Display summary of top N solutions only
        solutions_to_display = all_solutions[:num_solutions_display]
        display_multi_solution_summary(solutions_to_display, show_f1_breakdown=True,
                                       show_detailed_f1=show_detailed_f1,
                                       df=df, current_team_indices=current_team_indices)

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

        return all_solutions

    except FileNotFoundError:
        print(f"Error: Could not find {excel_file}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main_multi_transfer_optimiser(
        excel_file="Fantasy Premier League.xlsx",
        max_transfers=2,
        num_fixtures=6,
        fixture_weights=[1.0, 0.90, 0.80, 0.75, 0.70, 0.65],
        show_current_analysis=False,
        additional_budget=1.5,
        bench_weights=[1, 0.3, 0.3, 0.3, 0.3, 0.3],
        gk_bench_weights=[1, 0.2, 0.2, 0.2, 0.2, 0.2],
        compute_solutions=20,
        num_solutions_display=10,
        show_all_details=False,
        show_detailed_f1=False,
        max_defensive_players_per_team=2,
    )
