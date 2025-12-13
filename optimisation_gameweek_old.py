import pandas as pd
import pulp
import numpy as np


def load_current_team(excel_file, sheet_name='GW Teams'):
    """
    Load the current team from the rightmost column in the GW Teams sheet

    Parameters:
    - excel_file: Path to Excel file
    - sheet_name: Sheet name containing team data

    Returns:
    - current_team_names: List of current player names
    - gameweek: The gameweek number (column header)
    """
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


def analyze_current_team(excel_file, current_team_names, num_fixtures=6, fixture_weights=None,
                         players_sheet='Players', additional_budget=0.0):
    """
    Analyze the current team's expected points and composition

    Parameters:
    - excel_file: Path to Excel file
    - current_team_names: List of current player names
    - num_fixtures: Number of fixtures to analyze
    - fixture_weights: Weights for each fixture
    - players_sheet: Sheet name with player data
    - additional_budget: Any remaining budget beyond player values

    Returns:
    - current_team_df: DataFrame with current team players and their data
    - total_weighted_points: Total weighted expected points
    - analysis: Dictionary with team analysis
    """
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
    """
    Display analysis of the current team
    """
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


def optimise_transfers(excel_file, current_team_names, max_transfers=2, num_fixtures=5,
                       fixture_weights=None, players_sheet='Players', teams_sheet='GW Teams',
                       additional_budget=0.0, bench_weight=0.1, force_transfer_out=None):
    """
    Optimise transfers from current team with complex objective function

    Parameters:
    - excel_file: Path to Excel file
    - current_team_names: List of current player names
    - max_transfers: Maximum number of transfers allowed
    - num_fixtures: Number of fixtures to optimise for
    - fixture_weights: Weights for each fixture
    - players_sheet: Sheet name with player data
    - teams_sheet: Sheet name with team data
    - additional_budget: Any remaining budget beyond current player values
    - bench_weight: Weight for bench players in objective function
    - force_transfer_out: List of player names that must be transferred out

    Returns:
    - Optimisation results including transfers to make
    """
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    weights = fixture_weights[:num_fixtures]

    # Handle forced transfers
    if force_transfer_out is None:
        force_transfer_out = []

    print(f"\nOptimising transfers: max {max_transfers} transfers for {num_fixtures} fixtures")
    print(f"Using fixture weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional budget available: £{additional_budget:.1f}m")
    print(f"Bench weight: {bench_weight:.2f}")
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

    # Create optimisation problem
    prob = pulp.LpProblem("FPL_Transfer_Optimisation", pulp.LpMaximize)

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

    # 4. Bench players: binary variable for players who are in squad but not starting
    bench_vars = {}
    for fixture in fixtures:
        bench_vars[fixture] = {}
        for i in df.index:
            bench_vars[fixture][i] = pulp.LpVariable(f"bench_{fixture}_{i}", cat='Binary')

    # transfer_out_vars: 1 if current player is transferred out, 0 otherwise
    transfer_out_vars = {}
    for i in current_team_indices:
        transfer_out_vars[i] = pulp.LpVariable(f"transfer_out_{i}", cat='Binary')

    # transfer_in_vars: 1 if new player is transferred in, 0 otherwise
    transfer_in_vars = {}
    for i in df.index:
        if i not in current_team_indices:
            transfer_in_vars[i] = pulp.LpVariable(f"transfer_in_{i}", cat='Binary')

    # Objective function: maximie total weighted expected points from starting XIs + captain bonuses + bench value
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
            # Weighted bench value (smaller weight for bench players)
            objective_terms.append(
                df.loc[player_idx, fixture_col] * weight * bench_weight * bench_vars[fixture][player_idx])

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

    # NEW CONSTRAINT: Force specific players to be transferred out
    for forced_idx in forced_out_indices:
        prob += transfer_out_vars[forced_idx] == 1
        print(f"Forcing transfer out: {df.loc[forced_idx, 'Player Name']}")

    # Constraint 5: Maximum transfers (adjusted for forced transfers)
    total_transfers = []
    total_transfers.extend([transfer_out_vars[i] for i in current_team_indices])
    total_transfers.extend([transfer_in_vars[i] for i in transfer_in_vars])

    prob += pulp.lpSum(total_transfers) <= max_transfers * 2  # Each transfer involves out + in

    # Constraint 6: Equal transfers in and out
    transfers_out = pulp.lpSum([transfer_out_vars[i] for i in current_team_indices])
    transfers_in = pulp.lpSum([transfer_in_vars[i] for i in transfer_in_vars])
    prob += transfers_out == transfers_in

    # Constraint 7: Budget constraint
    # Cost = cost_of_players_staying + cost_of_new_transfers_in <= total_budget
    cost_terms = []

    # Add costs of players staying in team
    for i in current_team_indices:
        cost_terms.append(df.loc[i, 'Cost'] * (1 - transfer_out_vars[i]))

    # Add costs of new players transferred in
    for i in transfer_in_vars:
        cost_terms.append(df.loc[i, 'Cost'] * transfer_in_vars[i])

    prob += pulp.lpSum(cost_terms) <= total_budget

    # Constraint 8: Starting XI constraints for each fixture
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

    # Constraint 9: Captain constraints for each fixture
    for fixture in fixtures:
        # Exactly 1 captain per fixture
        prob += pulp.lpSum([captain_vars[fixture][i] for i in df.index]) == 1

        # Captain must be a starter
        for i in df.index:
            prob += captain_vars[fixture][i] <= starting_vars[fixture][i]

    # Constraint 10: Bench constraints for each fixture
    for fixture in fixtures:
        # Exactly 4 bench players per fixture (15 - 11)
        prob += pulp.lpSum([bench_vars[fixture][i] for i in df.index]) == 4

        # Bench players must be in squad but not starting
        for i in df.index:
            prob += bench_vars[fixture][i] <= squad_vars[i]
            prob += bench_vars[fixture][i] <= (1 - starting_vars[fixture][i])
            # If in squad and not starting, must be on bench
            prob += bench_vars[fixture][i] >= squad_vars[i] - starting_vars[fixture][i]

    # Constraint 11: Starting players must be in squad
    for fixture in fixtures:
        for i in df.index:
            prob += starting_vars[fixture][i] <= squad_vars[i]

    # Solve the problem
    print("Solving transfer optimisation...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Check solution status
    status = pulp.LpStatus[prob.status]
    print(f"Optimisation Status: {status}")

    if prob.status != pulp.LpStatusOptimal:
        print("No optimal solution found!")
        return None

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

        # Starting points
        for player_idx in starting_lineups[fixture]:
            total_starting_points += df.loc[player_idx, fixture_col] * weight

        # Captain bonus points
        if fixture in captains:
            captain_idx = captains[fixture]
            total_captain_points += df.loc[captain_idx, fixture_col] * weight

        # Bench points
        for player_idx in bench_players[fixture]:
            total_bench_points += df.loc[player_idx, fixture_col] * weight * bench_weight

    final_points = total_starting_points + total_captain_points + total_bench_points

    # Calculate current team points for comparison
    current_team_df, current_points, _ = analyze_current_team(excel_file, current_team_names,
                                                              num_fixtures, fixture_weights,
                                                              players_sheet, additional_budget)

    points_improvement = final_points - current_points
    num_transfers = len(transfers_out)

    return {
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
        'total_starting_points': total_starting_points,
        'total_captain_points': total_captain_points,
        'total_bench_points': total_bench_points,
        'current_points': current_points,
        'points_improvement': points_improvement,
        'num_transfers': num_transfers,
        'max_transfers': max_transfers,
        'weights': weights,
        'additional_budget': additional_budget,
        'fixtures': fixtures,
        'bench_weight': bench_weight,
        'forced_transfers_out': force_transfer_out,
        'forced_out_indices': forced_out_indices
    }


def get_top_replacements(df, player_to_replace, current_team_indices, max_cost_increase=5.0, top_n=5):
    """
    Get top replacement options for a specific player

    Parameters:
    - df: Player dataframe
    - player_to_replace: Series with player data to replace
    - current_team_indices: List of current team player indices
    - max_cost_increase: Maximum cost increase allowed for replacement
    - top_n: Number of top replacements to return

    Returns:
    - DataFrame with top replacement options
    """
    position = player_to_replace['Position']
    team = player_to_replace['Team']
    max_cost = player_to_replace['Cost'] + max_cost_increase

    # Filter potential replacements
    replacements = df[
        (df['Position'] == position) &  # Same position
        (df['Cost'] <= max_cost) &  # Within budget
        (~df.index.isin(current_team_indices))  # Not already in team
        ].copy()

    # Add cost difference and value metrics
    replacements['Cost_Diff'] = replacements['Cost'] - player_to_replace['Cost']
    replacements['Points_Diff'] = replacements['Weighted_Total_XP'] - player_to_replace['Weighted_Total_XP']
    replacements['Value'] = replacements['Weighted_Total_XP'] / replacements['Cost']
    replacements['Same_Team'] = replacements['Team'] == team

    # Sort by weighted points (primary) and value (secondary)
    replacements_sorted = replacements.sort_values(
        ['Weighted_Total_XP', 'Value'],
        ascending=[False, False]
    )

    return replacements_sorted.head(top_n)


def analyse_transfer_alternatives(excel_file, current_team_names, transfers_out_players,
                                  num_fixtures=6, fixture_weights=None, players_sheet='Players'):
    """
    Analyse alternative players for each transfer out

    Parameters:
    - excel_file: Path to Excel file
    - current_team_names: List of current player names
    - transfers_out_players: DataFrame of players being transferred out
    - num_fixtures: Number of fixtures analyzed
    - fixture_weights: Weights for each fixture
    - players_sheet: Sheet name with player data

    Returns:
    - Dictionary with alternatives for each transferred out player
    """
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    # Load player data
    df = pd.read_excel(excel_file, sheet_name=players_sheet)
    df.columns = df.columns.str.strip()

    # Calculate weighted total XP
    all_fixture_columns = ['F1 XP', 'F2 XP', 'F3 XP', 'F4 XP', 'F5 XP', 'F6 XP']
    fixture_columns = all_fixture_columns[:num_fixtures]
    weights = fixture_weights[:num_fixtures]

    df['Weighted_Total_XP'] = 0
    for i, fixture_col in enumerate(fixture_columns):
        weight = weights[i]
        df['Weighted_Total_XP'] += df[fixture_col] * weight

    # Get current team indices
    current_team_indices = []
    for player_name in current_team_names:
        matches = df[df['Player Name'].str.strip() == player_name.strip()]
        if len(matches) == 0:
            matches = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]
        if len(matches) > 0:
            current_team_indices.append(matches.index[0])

    # Get alternatives for each transfer out
    alternatives = {}
    for _, player_out in transfers_out_players.iterrows():
        player_name = player_out['Player Name']

        # Find the player in the main dataframe
        player_match = df[df['Player Name'].str.strip() == player_name.strip()]
        if len(player_match) == 0:
            player_match = df[df['Player Name'].str.contains(player_name.strip(), case=False, na=False)]

        if len(player_match) > 0:
            player_data = player_match.iloc[0]
            top_replacements = get_top_replacements(df, player_data, current_team_indices)
            alternatives[player_name] = top_replacements

    return alternatives


def display_transfer_alternatives(alternatives):
    """
    Display top alternatives for each transfer out
    """
    if not alternatives:
        return

    print(f"\nTOP REPLACEMENT OPTIONS:")
    print("=" * 120)

    for player_out, replacements in alternatives.items():
        print(f"\nAlternatives for {player_out}:")
        print("-" * 80)

        if len(replacements) == 0:
            print("  No suitable replacements found within budget constraints")
            continue

        print(f"{'Rank':<4} {'Player Name':<25} {'Team':<8} {'Cost':<8} {'Cost Diff':<10} "
              f"{'Points':<8} {'Pts Diff':<10} {'Value':<8}")
        print("-" * 80)

        for i, (_, replacement) in enumerate(replacements.iterrows(), 1):
            cost_diff = replacement['Cost_Diff']
            points_diff = replacement['Points_Diff']
            value = replacement['Value']
            same_team = "✓" if replacement['Same_Team'] else ""

            print(f"{i:<4} {replacement['Player Name']:<25} {replacement['Team']:<8} "
                  f"£{replacement['Cost']:.1f}m{'':<3} {cost_diff:+.1f}m{'':<5} "
                  f"{replacement['Weighted_Total_XP']:.2f}{'':<4} {points_diff:+.2f}{'':<6} "
                  f"{value:.3f}{'':<4} {same_team}")


def display_starting_lineup(transfer_results, fixture_num=1):
    """
    Display the optimal starting XI for a specific fixture

    Parameters:
    - transfer_results: Results from optimisation
    - fixture_num: Which fixture to display (1-based)
    """
    if transfer_results is None:
        print("No transfer results available")
        return

    fixtures = transfer_results['fixtures']
    if fixture_num > len(fixtures):
        print(f"Fixture {fixture_num} not available. Only {len(fixtures)} fixtures analyzed.")
        return

    fixture = fixtures[fixture_num - 1]  # Convert to 0-based
    starting_lineup = transfer_results['starting_lineups'][fixture]
    captain_idx = transfer_results['captains'].get(fixture)
    bench = transfer_results['bench_players'][fixture]

    # Get player data
    final_squad = transfer_results['final_squad']

    print(f"\n" + "=" * 80)
    print(f"OPTIMAL STARTING XI - FIXTURE {fixture_num} ({fixture})")
    print("=" * 80)

    # Get starting players data
    starting_players = final_squad[final_squad.index.isin(starting_lineup)].copy()
    bench_players_data = final_squad[final_squad.index.isin(bench)].copy()

    # Add captain indicator
    starting_players['Is_Captain'] = starting_players.index == captain_idx

    # Sort by position and points
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

    # Show bench
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


def display_transfer_results(transfer_results, num_fixtures, show_alternatives=True,
                             excel_file="Fantasy Premier League.xlsx", current_team_names=None,
                             show_starting_xi=True):
    """
    Enhanced display function that includes transfer alternatives, budget info, and starting XI
    """
    if transfer_results is None:
        print("No transfer results to display!")
        return

    # Show alternatives for transfers out
    if show_alternatives and len(transfer_results['transfers_out']) > 0 and current_team_names:
        try:
            alternatives = analyse_transfer_alternatives(
                excel_file,
                current_team_names,
                transfer_results['transfers_out'],
                num_fixtures,
                transfer_results['weights']
            )
            display_transfer_alternatives(alternatives)
        except Exception as e:
            print(f"\nNote: Could not generate transfer alternatives: {e}")

    # Show complete final squad
    print(f"\nFINAL SQUAD AFTER TRANSFERS:")
    print("-" * 80)
    final_squad_sorted = transfer_results['final_squad'].sort_values(['Position', 'Weighted_Total_XP'],
                                                                     ascending=[True, False])

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = final_squad_sorted[final_squad_sorted['Position'] == pos]
        if len(pos_players) > 0:
            print(f"\n{pos} ({len(pos_players)}):")
            for _, player in pos_players.iterrows():
                print(f"  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                      f"{player['Weighted_Total_XP']:.2f} pts")

    # Show starting XI for next gameweek (fixture 1)
    if show_starting_xi and 'starting_lineups' in transfer_results:
        display_starting_lineup(transfer_results, fixture_num=1)

    # Transfers out
    if len(transfer_results['transfers_out']) > 0:
        print(f"\nTRANSFERS OUT ({len(transfer_results['transfers_out'])}):")
        print("-" * 80)
        for _, player in transfer_results['transfers_out'].iterrows():
            print(f"  OUT: {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                  f"{player['Weighted_Total_XP']:.2f} pts {player['F1 XP']:.2f} pts")

    # Transfers in
    if len(transfer_results['transfers_in']) > 0:
        print(f"\nTRANSFERS IN ({len(transfer_results['transfers_in'])}):")
        print("-" * 80)
        for _, player in transfer_results['transfers_in'].iterrows():
            print(f"  IN:  {player['Player Name']:<25} {player['Team']:<8} £{player['Cost']:.1f}m  "
                  f"{player['Weighted_Total_XP']:.2f} pts {player['F1 XP']:.2f} pts")


def main_transfer_optimiser(excel_file="Fantasy Premier League.xlsx", max_transfers=2, num_fixtures=5,
                            fixture_weights=None, show_current_analysis=True, show_alternatives=True,
                            additional_budget=0.0, bench_weight=0.1, show_starting_lineup=True,
                            force_transfer_out=None):
    """
    Enhanced main function with dynamic budget support and complex objective function

    Parameters:
    - excel_file: Path to Excel file
    - max_transfers: Maximum transfers allowed
    - num_fixtures: Number of fixtures to optimie for
    - fixture_weights: Weights for each fixture
    - show_current_analysis: Whether to show current team analysis
    - show_alternatives: Whether to show top 5 alternatives for transfers out
    - additional_budget: Any remaining budget beyond current player values
    - bench_weight: Weight for bench players in objective function
    - show_starting_xi: Whether to show the starting XI for next gameweek
    """
    # Set default weights
    if fixture_weights is None:
        fixture_weights = [1.0, 0.85, 0.7, 0.55, 0.4]

    weights = fixture_weights[:num_fixtures]

    print(f"FPL TRANSFER OPTIMISER")
    print(f"Max Transfers: {max_transfers}")
    print(f"Fixtures: {num_fixtures}")
    print(f"Weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Additional Budget: £{additional_budget:.1f}m")
    print(f"Bench Weight: {bench_weight:.2f}")

    try:
        # Load current team
        current_team_names, gameweek = load_current_team(excel_file)

        if show_current_analysis:
            # Analyze current team
            current_team_df, current_points, analysis = analyze_current_team(
                excel_file, current_team_names, num_fixtures, fixture_weights,
                'Players', additional_budget
            )

            # Display current team analysis
            display_current_team_analysis(current_team_df, analysis, num_fixtures, weights)

        # Optimise transfers
        transfer_results = optimise_transfers(
            excel_file, current_team_names, max_transfers, num_fixtures,
            fixture_weights, 'Players', 'GW Teams', additional_budget, bench_weight,
            force_transfer_out
        )

        # Display results with alternatives and starting XI
        display_transfer_results(transfer_results, num_fixtures, show_alternatives,
                                 excel_file, current_team_names, show_starting_lineup)

        return transfer_results

    except FileNotFoundError:
        print(f"Error: Could not find {excel_file}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main_transfer_optimiser(
        excel_file="Fantasy Premier League.xlsx",
        max_transfers=2,
        num_fixtures=6,
        fixture_weights=[1.0, 0.9, 0.75, 0.7, 0.65, 0.6],
        show_current_analysis=True,
        show_alternatives=True,
        additional_budget=3.5,
        bench_weight=0.3,
        show_starting_lineup=True,
        # force_transfer_out=["Benjamin Sesko"],
    )
