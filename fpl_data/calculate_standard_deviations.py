import pandas as pd
import numpy as np
import os
import glob

REPO_PATH = "FPL-Core-Insights"
SEASON = "2025-2026"
TOURNAMENT = "Premier League"

BASE_PATH = os.path.join(REPO_PATH, "data", SEASON, "By Tournament", TOURNAMENT)


def load_all_gameweeks():
    gw_dirs = sorted(glob.glob(os.path.join(BASE_PATH, "GW*")))

    if not gw_dirs:
        print(f"No gameweek directories found in: {BASE_PATH}")
        print("Make sure FPL_CORE_INSIGHTS_PATH is set or the repo is cloned to ~/FPL-Core-Insights")
        return None, None

    print(f"Found {len(gw_dirs)} gameweek directories")

    all_match_stats = []
    players_df = None

    for gw_dir in gw_dirs:
        gw_name = os.path.basename(gw_dir)

        match_stats_path = os.path.join(gw_dir, "player_gameweek_stats.csv")
        players_path = os.path.join(gw_dir, "players.csv")

        if not os.path.exists(match_stats_path):
            print(f"  Skipping {gw_name}: no player_gameweek_stats.csv")
            continue

        try:
            match_stats = pd.read_csv(match_stats_path)
            match_stats["gameweek"] = gw_name
            all_match_stats.append(match_stats)

            # Use the latest players.csv for position data
            if os.path.exists(players_path):
                players_df = pd.read_csv(players_path)
        except Exception as e:
            print(f"  Error reading {gw_name}: {e}")
            continue

    if not all_match_stats:
        print("No match stats data loaded!")
        return None, None

    combined = pd.concat(all_match_stats, ignore_index=True)
    print(f"Loaded {len(combined)} player-match observations across {len(all_match_stats)} gameweeks")

    return combined, players_df


def analyse_position(match_stats, players_df, position):
    # Get player IDs for this position
    pos_players = players_df[players_df["position"] == position]["player_id"]

    # Filter to this position
    df = match_stats[match_stats["id"].isin(pos_players)].copy()
    print(f"\n{'=' * 60}")
    print(f"Position: {position}")
    print(f"{'=' * 60}")
    print(f"Total player-match observations: {len(df)}")

    # Filter to players who played 90 minutes
    df_90 = df[df["minutes"] == 90].copy()
    print(f"Observations with exactly 90 minutes played: {len(df_90)}")

    # Check for required columns
    required_cols = ["defensive_contribution"]

    missing_cols = [c for c in required_cols if c not in df_90.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        return None

    # Drop rows where any defensive stat is NaN
    df_90 = df_90.dropna(subset=required_cols)
    print(f"Observations after dropping NaN defensive stats: {len(df_90)}")

    if len(df_90) == 0:
        print("No valid observations!")
        return None

    # Calculate statistics
    mean_val = df_90["defensive_contribution"].mean()
    sd_val = df_90["defensive_contribution"].std(ddof=1)  # sample SD
    median_val = df_90["defensive_contribution"].median()
    n_obs = len(df_90)
    n_players = df_90["id"].nunique()

    # Threshold for FPL points
    threshold = 10 if position == "Defender" else 12
    pct_above = (df_90["defensive_contribution"] >= threshold).mean() * 100

    print(f"\nResults (per 90 minutes):")
    print(f"  Unique players:     {n_players}")
    print(f"  Observations:       {n_obs}")
    print(f"  Mean:               {mean_val:.2f}")
    print(f"  Median:             {median_val:.2f}")
    print(f"  Std Dev:            {sd_val:.2f}")
    print(f"  Min:                {df_90['defensive_contribution'].min():.0f}")
    print(f"  Max:                {df_90['defensive_contribution'].max():.0f}")
    print(f"  % >= {threshold} (FPL pts):  {pct_above:.1f}%")

    # Show distribution
    print(f"\n  Distribution:")
    for pct in [10, 25, 50, 75, 90]:
        val = np.percentile(df_90["defensive_contribution"], pct)
        print(f"    {pct}th percentile:  {val:.1f}")

    # Top players by average
    player_avg = (
        df_90.groupby("id")
        .agg(
            mean_dc=("defensive_contribution", "mean"),
            sd_dc=("defensive_contribution", "std"),
            n_matches=("defensive_contribution", "count"),
        )
        .sort_values("mean_dc", ascending=False)
    )

    return {
        "position": position,
        "n_players": n_players,
        "n_observations": n_obs,
        "mean": mean_val,
        "sd": sd_val,
        "median": median_val,
        "pct_above_threshold": pct_above,
        "threshold": threshold,
    }


def main():
    print("Loading FPL-Core-Insights data...")
    print(f"Path: {BASE_PATH}\n")

    match_stats, players_df = load_all_gameweeks()

    if match_stats is None or players_df is None:
        return

    # Show available columns for debugging
    print(f"\nplayermatchstats columns: {sorted(match_stats.columns.tolist())}")
    print(f"players columns: {sorted(players_df.columns.tolist())}")

    results = {}
    for position in ["Defender", "Midfielder"]:
        result = analyse_position(match_stats, players_df, position)
        if result:
            results[position] = result

    # Summary comparison
    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Metric':<30s} {'DEF':>10s} {'MID':>10s}")
        print(f"{'-' * 50}")
        d, m = results["Defender"], results["Midfielder"]
        print(f"{'Observations':<30s} {d['n_observations']:>10d} {m['n_observations']:>10d}")
        print(f"{'Unique players':<30s} {d['n_players']:>10d} {m['n_players']:>10d}")
        print(f"{'Mean per 90':<30s} {d['mean']:>10.2f} {m['mean']:>10.2f}")
        print(f"{'SD per 90':<30s} {d['sd']:>10.2f} {m['sd']:>10.2f}")
        print(f"{'Median per 90':<30s} {d['median']:>10.2f} {m['median']:>10.2f}")
        print(f"{'FPL threshold':<30s} {d['threshold']:>10d} {m['threshold']:>10d}")
        print(f"{'% above threshold':<30s} {d['pct_above_threshold']:>9.1f}% {m['pct_above_threshold']:>9.1f}%")


if __name__ == "__main__":
    main()
