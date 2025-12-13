import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = """0.26,0.0,0.04,0.02,A,0.50
0.01,0.01,0.01,0.0,H,0.40
0.0,0.08,0.33,0.0,A,0.11
0.0,0.24,0.0,0.09,A,0.23
0.0,0.04,0.0,0.07,H,0.53
0.0,0.68,0.02,0.01,A,0.15
0.10,0.0,0.0,0.09,H,0.53
0.0,0.09,0.10,0.0,A,0.22
0.0,0.10,0.0,0.40,A,0.31
0.0,0.07,0.0,0.04,A,0.22
0.0,0.40,0.0,0.10,H,0.40
0.33,0.0,0.0,0.08,H,0.75
0.21,0.05,0.0,0.22,A,0.63
0.04,0.02,0.26,0.0,H,0.25
0.01,0.0,0.01,0.01,A,0.33
0.0,0.09,0.0,0.24,H,0.51
0.02,0.01,0.0,0.68,H,0.62
0.0,0.66,0.0,0.15,H,0.28
0.0,0.15,0.0,0.66,A,0.43
0.0,0.22,0.21,0.05,H,0.15
0.35,0.0,0.35,0.0,A,0.26
0.01,0.02,0.0,0.10,H,0.52
0.0,0.09,0.04,0.01,A,0.21
0.0,0.25,0.0,0.61,A,0.37
0.0,0.06,0.17,0.05,H,0.23
0.0,0.70,0.03,0.03,A,0.11
0.12,0.0,0.0,0.10,H,0.63
0.0,0.10,0.01,0.02,A,0.22
0.0,0.10,0.0,0.38,A,0.33
0.0,0.10,0.12,0.0,A,0.17
0.0,0.29,0.01,0.01,H,0.27
0.35,0.0,0.35,0.0,H,0.49
0.17,0.05,0.0,0.06,A,0.55
0.03,0.03,0.0,0.70,H,0.72
0.01,0.01,0.0,0.29,A,0.49
0.01,0.09,0.0,0.35,H,0.57
0.04,0.01,0.0,0.09,H,0.57
0.0,0.61,0.0,0.25,H,0.36
0.0,0.35,0.01,0.09,A,0.19
0.0,0.38,0.0,0.10,H,0.39
0.35,0.0,0.01,0.09,H,0.68
0.01,0.02,0.0,0.10,A,0.41
0.0,0.09,0.0,0.06,H,0.40
0.0,0.25,0.12,0.0,H,0.21
0.0,0.06,0.0,0.09,A,0.35
0.0,0.70,0.35,0.0,H,0.10
0.12,0.0,0.0,0.25,A,0.56
0.0,0.10,0.0,0.61,H,0.57
0.0,0.10,0.01,0.02,H,0.31
0.0,0.10,0.0,0.29,H,0.52
0.0,0.29,0.0,0.10,A,0.24
0.35,0.0,0.0,0.70,A,0.74
0.17,0.05,0.03,0.03,H,0.55
0.03,0.03,0.17,0.05,A,0.23
0.01,0.01,0.0,0.38,H,0.66
0.01,0.09,0.35,0.0,A,0.14
0.04,0.01,0.0,0.35,A,0.51
0.0,0.61,0.0,0.10,A,0.20
0.0,0.35,0.04,0.01,H,0.25
0.0,0.38,0.01,0.01,A,0.15"""

# Parse the data
rows = [line.split(',') for line in data.strip().split('\n')]
df = pd.DataFrame(rows, columns=['Winner_Prob', 'Relegation_Prob', 'Opponent_Winner_Prob',
                                 'Opponent_Relegation_Prob', 'Venue', 'Match_Win_Prob'])

# Convert to numeric (except venue) - data is already in decimal format
for col in df.columns:
    if col != 'Venue':
        df[col] = pd.to_numeric(df[col])

# Convert venue to binary (H=1, A=0)
df['Venue_Home'] = (df['Venue'] == 'H').astype(int)

# Prepare features and target - data is already in decimal format
feature_columns = ['Winner_Prob', 'Relegation_Prob', 'Opponent_Winner_Prob',
                   'Opponent_Relegation_Prob', 'Venue_Home']
X = df[feature_columns]
y = df['Match_Win_Prob']  # Already in decimal format


class MatchWinPredictor:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self.feature_names = None

    def fit(self, X, y):
        # Create polynomial features
        X_poly = self.poly_features.fit_transform(X)
        self.feature_names = self.poly_features.get_feature_names_out(X.columns)

        # Fit the model
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)

    def get_formula(self, X_columns):
        """Generate the polynomial formula as a string"""
        coefficients = self.model.coef_
        intercept = self.model.intercept_

        # Create readable feature names mapping
        feature_mapping = {
            'Winner_Prob': 'WP',
            'Relegation_Prob': 'RP',
            'Opponent_Winner_Prob': 'OWP',
            'Opponent_Relegation_Prob': 'ORP',
            'Venue_Home': 'VH'
        }

        formula = f"Match_Win_Prob = {intercept:.6f}"

        for i, (coef, feature_name) in enumerate(zip(coefficients, self.feature_names)):
            if abs(coef) > 1e-10:  # Only include non-zero coefficients
                # Clean up the feature name for readability
                readable_feature = feature_name
                for original, short in feature_mapping.items():
                    readable_feature = readable_feature.replace(original, short)

                # Fix spaces in interaction terms
                readable_feature = readable_feature.replace(' ', ' * ')

                sign = " + " if coef >= 0 else " - "
                formula += f"{sign}{abs(coef):.6f} * {readable_feature}"

        return formula

    def get_simplified_formula(self, X_columns):
        """Generate a more readable version with variable explanations"""
        coefficients = self.model.coef_
        intercept = self.model.intercept_

        print("Variable Legend:")
        print("WP = Winner_Prob (team's title winner probability)")
        print("RP = Relegation_Prob (team's relegation probability)")
        print("OWP = Opponent_Winner_Prob (opponent's title winner probability)")
        print("ORP = Opponent_Relegation_Prob (opponent's relegation probability)")
        print("VH = Venue_Home (1 if home, 0 if away)")
        print("\nSimplified Formula:")

        formula_parts = [f"{intercept:.6f}"]

        for i, (coef, feature_name) in enumerate(zip(coefficients, self.feature_names)):
            if abs(coef) > 1e-10:
                # Simplify feature names
                readable_feature = feature_name
                readable_feature = readable_feature.replace('Winner_Prob', 'WP')
                readable_feature = readable_feature.replace('Relegation_Prob', 'RP')
                readable_feature = readable_feature.replace('Opponent_Winner_Prob', 'OWP')
                readable_feature = readable_feature.replace('Opponent_Relegation_Prob', 'ORP')
                readable_feature = readable_feature.replace('Venue_Home', 'VH')
                readable_feature = readable_feature.replace(' ', '·')  # Use · for multiplication

                if coef >= 0:
                    formula_parts.append(f" + {coef:.6f}·{readable_feature}")
                else:
                    formula_parts.append(f" - {abs(coef):.6f}·{readable_feature}")

        return "Match_Win_Prob = " + "".join(formula_parts)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2


# Try different polynomial degrees
degrees = [1]
results = {}

print("Testing different polynomial degrees:")
print("=" * 50)

for degree in degrees:
    predictor = MatchWinPredictor(degree=degree)
    predictor.fit(X, y)

    # Evaluate on the same data (in practice, you'd use train/test split)
    mse, r2 = predictor.evaluate(X, y)
    results[degree] = {'predictor': predictor, 'mse': mse, 'r2': r2}

    print(f"\nDegree {degree} Polynomial:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.6f}")

# Select the best model based on R²
best_degree = max(results.keys(), key=lambda x: results[x]['r2'])
best_predictor = results[best_degree]['predictor']

print(f"\n" + "=" * 50)
print(f"BEST MODEL: Degree {best_degree} Polynomial")
print(f"R² Score: {results[best_degree]['r2']:.4f}")
print("=" * 50)

# Generate the formula
formula = best_predictor.get_formula(X.columns)
print(f"\nPOLYNOMIAL FORMULA:")
print(formula)


# Function to make predictions
def predict_match_win_probability(winner_prob, relegation_prob, opponent_winner_prob,
                                  opponent_relegation_prob, venue_home):
    """
    Predict match win probability using the trained model

    Parameters:
    - winner_prob: Team's title winner probability (0.0-1.0 decimal)
    - relegation_prob: Team's relegation probability (0.0-1.0 decimal)
    - opponent_winner_prob: Opponent's title winner probability (0.0-1.0 decimal)
    - opponent_relegation_prob: Opponent's relegation probability (0.0-1.0 decimal)
    - venue_home: 1 if home game, 0 if away game

    Returns:
    - Match win probability as decimal (0.0-1.0)
    """
    input_data = np.array([[winner_prob, relegation_prob, opponent_winner_prob,
                            opponent_relegation_prob, venue_home]])
    prediction = best_predictor.predict(input_data)[0]
    return prediction  # Already in decimal format


# Example predictions
print(f"\n" + "=" * 50)
print("EXAMPLE PREDICTIONS:")
print("=" * 50)

# Example 1: Strong team at home vs weak opponent
example1 = predict_match_win_probability(0.25, 0.0, 0.02, 0.15, 1)  # Home
print(
    f"Strong team (0.25 title odds, 0.0 relegation) vs weak opponent (0.02 title, 0.15 relegation) at HOME: {example1:.3f}")

# Example 2: Same teams, but away
example2 = predict_match_win_probability(0.25, 0.0, 0.02, 0.15, 0)  # Away
print(f"Same matchup but AWAY: {example2:.3f}")

# Example 3: Even matchup
example3 = predict_match_win_probability(0.10, 0.02, 0.08, 0.03, 1)  # Home
print(f"Even matchup (0.10 vs 0.08 title odds) at HOME: {example3:.3f}")

print(f"\n" + "=" * 50)
print("HOW TO USE THE PREDICTOR:")
print("=" * 50)
print(
    "Call: predict_match_win_probability(winner_prob, relegation_prob, opponent_winner_prob, opponent_relegation_prob, venue_home)")
print("- All probability inputs should be 0.0-1.0 (decimal fractions)")
print("- venue_home: 1 for home games, 0 for away games")
print("- Returns: Match win probability as decimal fraction (0.0-1.0)")

# Visualize actual vs predicted
y_pred = best_predictor.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Match Win Probability (decimal)')
plt.ylabel('Predicted Match Win Probability (decimal)')
plt.title(
    f'Actual vs Predicted Match Win Probabilities\n(Degree {best_degree} Polynomial, R² = {results[best_degree]["r2"]:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance analysis (for degree 1 only)
if best_degree == 1:
    feature_importance = abs(best_predictor.model.coef_)
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance (Linear Model)')
    plt.tight_layout()
    plt.show()
