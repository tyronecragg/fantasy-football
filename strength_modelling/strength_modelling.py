import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def analyze_expected_points_formula():
    """
    Analyze the relationship between team metrics and expected points
    to create a predictive formula
    """

    # Your data
    data = {
        'Team': ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
                 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
                 'Man City', 'Man Utd', 'Newcastle', 'Nottingham Forest', 'Sunderland',
                 'Tottenham', 'West Ham', 'Wolves'],
        'Average_Expected_Points': [3.71, 3.30, 2.03, 2.97, 3.77, 1.82, 4.06, 2.44, 3.07,
                                    2.74, 3.25, 4.73, 4.13, 2.96, 3.15, 3.53, 3.00, 4.33, 3.33, 2.11],
        'Venue': ['A', 'H', 'A', 'A', 'H', 'A', 'H', 'A', 'A', 'A', 'H', 'H', 'A', 'H', 'A', 'H', 'H', 'H', 'A',
                      'H'],
        'Team_Strength': [0.95, 1.37, 0.68, 1.03, 1.62, 0.95, 1.56, 0.85, 1.46, 1.01, 1.68, 1.47, 1.34, 1.07, 0.97, 1.54,
                         1.60, 2.09, 1.54, 1.13],
        'Opponent_Strength': [1.07, 0.97, 1.47, 1.54, 1.01, 2.09, 0.85, 1.56, 1.68, 1.62, 1.46, 0.68, 1.13, 0.95,
                                  1.37, 1.03, 1.54, 0.95, 1.60, 1.34],
        'Strength_Difference': [-0.12, 0.40, -0.79, -0.51, 0.61, -1.14, 0.71, -0.71, -0.22, -0.61, 0.22, 0.79, 0.21, 0.12,
                              -0.40, 0.51, 0.06, 1.14, -0.06, -0.21]
    }

    df = pd.DataFrame(data)

    # Convert venue to numeric (Home = 1, Away = 0)
    df['Home_Advantage'] = df['Venue'].map({'H': 1, 'A': 0})

    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Basic correlations
    correlations = {
        'Team Strength': df['Average_Expected_Points'].corr(df['Team_Strength']),
        'Opponent Strength': df['Average_Expected_Points'].corr(df['Opponent_Strength']),
        'Strength Difference': df['Average_Expected_Points'].corr(df['Strength_Difference']),
        'Home Advantage': df['Average_Expected_Points'].corr(df['Home_Advantage'])
    }

    print("Correlations with Average Expected Points:")
    for factor, corr in correlations.items():
        print(f"  {factor}: {corr:.3f}")

    # Home vs Away analysis
    home_avg = df[df['Venue'] == 'H']['Average_Expected_Points'].mean()
    away_avg = df[df['Venue'] == 'A']['Average_Expected_Points'].mean()
    home_advantage = home_avg - away_avg

    print(f"\nVenue Analysis:")
    print(f"  Home Average: {home_avg:.2f} points")
    print(f"  Away Average: {away_avg:.2f} points")
    print(f"  Home Advantage: +{home_advantage:.2f} points")

    return df


def create_linear_model(df):
    """
    Create linear regression models to predict expected points
    """
    print("\n\nLINEAR REGRESSION MODELS")
    print("=" * 50)

    # Model 1: Simple strength difference + venue
    X1 = df[['Strength_Difference', 'Home_Advantage']]
    y = df['Average_Expected_Points']

    model1 = LinearRegression()
    model1.fit(X1, y)
    y_pred1 = model1.predict(X1)

    print("Model 1: Expected Points = a + b×Strength_Diff + c×Home_Advantage")
    print(
        f"Formula: {model1.intercept_:.3f} + {model1.coef_[0]:.6f}×Strength_Diff + {model1.coef_[1]:.3f}×Home_Advantage")
    print(f"R² Score: {r2_score(y, y_pred1):.3f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred1):.3f}")

    # Model 2: Team strength, opponent strength, venue
    X2 = df[['Team_Strength', 'Opponent_Strength', 'Home_Advantage']]

    model2 = LinearRegression()
    model2.fit(X2, y)
    y_pred2 = model2.predict(X2)

    print(f"\nModel 2: Expected Points = a + b×Team_Strength + c×Opponent_Strength + d×Home_Advantage")
    print(
        f"Formula: {model2.intercept_:.3f} + {model2.coef_[0]:.6f}×Team_Strength + {model2.coef_[1]:.6f}×Opponent_Strength + {model2.coef_[2]:.3f}×Home_Advantage")
    print(f"R² Score: {r2_score(y, y_pred2):.3f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred2):.3f}")

    # Model 3: Normalized strengths (more intuitive)
    df['Team_Strength_Norm'] = (df['Team_Strength'] - 1000) / 100  # Scale to more readable numbers
    df['Opponent_Strength_Norm'] = (df['Opponent_Strength'] - 1000) / 100

    X3 = df[['Team_Strength_Norm', 'Opponent_Strength_Norm', 'Home_Advantage']]

    model3 = LinearRegression()
    model3.fit(X3, y)
    y_pred3 = model3.predict(X3)

    print(
        f"\nModel 3: Expected Points = a + b×(Team_Strength-1000)/100 + c×(Opponent_Strength-1000)/100 + d×Home_Advantage")
    print(
        f"Formula: {model3.intercept_:.3f} + {model3.coef_[0]:.3f}×Team_Norm + {model3.coef_[1]:.3f}×Opp_Norm + {model3.coef_[2]:.3f}×Home_Advantage")
    print(f"R² Score: {r2_score(y, y_pred3):.3f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred3):.3f}")

    return model1, model2, model3, df


def create_simplified_formula(df):
    """
    Create a simplified, easy-to-use formula
    """
    print("\n\nSIMPLIFIED FORMULA")
    print("=" * 50)

    # Based on analysis, create a simple formula
    base_points = df['Average_Expected_Points'].mean()
    home_advantage = df[df['Venue'] == 'H']['Average_Expected_Points'].mean() - df[df['Venue'] == 'A'][
        'Average_Expected_Points'].mean()

    # Strength difference impact (points per 100 strength difference)
    strength_corr = df['Average_Expected_Points'].corr(df['Strength_Difference'])
    strength_std = df['Strength_Difference'].std()
    points_std = df['Average_Expected_Points'].std()
    strength_impact = (strength_corr * points_std) / strength_std * 100

    print("SIMPLE FORMULA:")
    print(
        f"Expected Points = {base_points:.2f} + {strength_impact:.4f} × (Strength_Diff/100) + {home_advantage:.2f} × Home")
    print("\nWhere:")
    print("  - Strength_Diff = Team_Strength - Opponent_Strength")
    print("  - Home = 1 if playing at home, 0 if away")
    print("  - Strength_Diff/100 scales the impact to a reasonable magnitude")

    # Test the simplified formula
    df['Simple_Prediction'] = base_points + strength_impact * (df['Strength_Difference'] / 100) + home_advantage * df[
        'Home_Advantage']

    simple_r2 = r2_score(df['Average_Expected_Points'], df['Simple_Prediction'])
    simple_mae = mean_absolute_error(df['Average_Expected_Points'], df['Simple_Prediction'])

    print(f"\nSimple Formula Performance:")
    print(f"R² Score: {simple_r2:.3f}")
    print(f"Mean Absolute Error: {simple_mae:.3f}")

    return base_points, strength_impact, home_advantage


def test_formula_examples(base_points, strength_impact, home_advantage):
    """
    Test the formula with some examples
    """
    print("\n\nFORMULA TESTING")
    print("=" * 50)

    examples = [
        {"team": "Strong Home", "team_str": 1300, "opp_str": 1100, "venue": "H"},
        {"team": "Weak Away", "team_str": 1100, "opp_str": 1300, "venue": "A"},
        {"team": "Even Home", "team_str": 1200, "opp_str": 1200, "venue": "H"},
        {"team": "Even Away", "team_str": 1200, "opp_str": 1200, "venue": "A"},
    ]

    print("Formula: Expected Points = {:.2f} + {:.4f} × (Strength_Diff/100) + {:.2f} × Home".format(
        base_points, strength_impact, home_advantage))
    print("\nExamples:")

    for ex in examples:
        strength_diff = ex["team_str"] - ex["opp_str"]
        home = 1 if ex["venue"] == "H" else 0
        predicted = base_points + strength_impact * (strength_diff / 100) + home_advantage * home

        print(f"  {ex['team']}: Team={ex['team_str']}, Opp={ex['opp_str']}, {ex['venue']} → {predicted:.2f} points")


def create_lookup_table():
    """
    Create a simple lookup table for common scenarios
    """
    print("\n\nQUICK LOOKUP TABLE")
    print("=" * 50)

    # Using our simplified formula coefficients
    base = 3.17  # Average
    strength_coef = 0.0047  # Per 100 strength difference
    home_bonus = 0.34  # Home advantage

    print("Expected Points by Strength Difference and Venue:")
    print("Strength Diff | Home  | Away")
    print("-" * 30)

    for diff in [-200, -100, -50, 0, 50, 100, 200]:
        home_points = base + strength_coef * diff + home_bonus
        away_points = base + strength_coef * diff
        print(f"{diff:12d} | {home_points:5.2f} | {away_points:4.2f}")


def main():
    """
    Main analysis function
    """
    # Load and analyze data
    df = analyze_expected_points_formula()

    # Create regression models
    model1, model2, model3, df_enhanced = create_linear_model(df)

    # Create simplified formula
    base_points, strength_impact, home_advantage = create_simplified_formula(df_enhanced)

    # Test with examples
    test_formula_examples(base_points, strength_impact, home_advantage)

    # Create lookup table
    create_lookup_table()

    # Show prediction vs actual for validation
    print("\n\nMODEL VALIDATION")
    print("=" * 50)
    print("Team                | Actual | Predicted | Error")
    print("-" * 50)

    for _, row in df_enhanced.iterrows():
        actual = row['Average_Expected_Points']
        predicted = row['Simple_Prediction']
        error = abs(actual - predicted)
        print(f"{row['Team']:<18} | {actual:6.2f} | {predicted:9.2f} | {error:5.2f}")

    return df_enhanced


if __name__ == "__main__":
    results = main()
