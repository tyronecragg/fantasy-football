import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Data from the FPL analysis
data = {
    'Average_Expected_Points': [3.71, 3.30, 2.03, 2.97, 3.77, 1.82, 4.06, 2.44, 3.07, 2.74,
                               3.25, 4.73, 4.13, 2.96, 3.15, 3.53, 3.00, 4.33, 3.33, 2.11],
    'GW1_Difficulty': [63, 46, 10, 35, 73, 4, 89, 11, 76, 27, 24, 90, 95, 37, 54, 65, 23, 96, 77, 5],
    'GW1_Opponent_Difficulty': [37, 54, 90, 65, 27, 96, 11, 89, 24, 73, 76, 10, 5, 63, 46, 35, 77, 4, 23, 95],
    'GW1_Difficulty_Difference': [25, -8, -81, -31, 46, -92, 77, -77, 51, -46, -51, 81, 90, -25, 8, 31, -55, 92, 55, -90],
    'GW1_Venue': ['A', 'H', 'A', 'A', 'H', 'A', 'H', 'A', 'A', 'A', 'H', 'H', 'A', 'H', 'A', 'H', 'H', 'H', 'A', 'H']
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode venue (Home=1, Away=0)
venue_encoder = LabelEncoder()
df['GW1_Venue'] = venue_encoder.fit_transform(df['GW1_Venue'])
print("Venue encoding:", dict(zip(venue_encoder.classes_, venue_encoder.transform(venue_encoder.classes_))))
print()

# Display basic statistics
print("Dataset Overview:")
print(df.describe())
print()

# Correlation matrix
print("Correlation Matrix with Target Variable:")
correlation_matrix = df.corr()
target_correlations = correlation_matrix['Average_Expected_Points'].sort_values(key=abs, ascending=False)
print(target_correlations)
print()

# Prepare features and target
X = df[['GW1_Difficulty', 'GW1_Opponent_Difficulty', 'GW1_Difficulty_Difference', 'GW1_Venue']]
y = df['Average_Expected_Points']

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print()

# Fit multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)

print("=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)

print(f"Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print()

print("Model Coefficients:")
print(f"Intercept: {model.intercept_:.4f}")
feature_names = ['GW1_Difficulty', 'GW1_Opponent_Difficulty', 'GW1_Difficulty_Difference', 'GW1_Venue']
for i, (feature, coef) in enumerate(zip(feature_names, model.coef_)):
    print(f"{feature}: {coef:.4f}")
print()

# Create prediction equation
print("Prediction Equation:")
equation = f"Expected_Points = {model.intercept_:.3f}"
for feature, coef in zip(feature_names, model.coef_):
    sign = "+" if coef >= 0 else ""
    if feature == 'GW1_Venue':
        equation += f" {sign}{coef:.3f} × Home"
    elif feature == 'GW1_Difficulty':
        equation += f" {sign}{coef:.3f} × Difficulty"
    elif feature == 'GW1_Opponent_Difficulty':
        equation += f" {sign}{coef:.3f} × Opponent_Difficulty"
    elif feature == 'GW1_Difficulty_Difference':
        equation += f" {sign}{coef:.3f} × Difficulty_Difference"

print(equation)
print("\nWhere:")
print("- Home = 1 if home fixture, 0 if away")
print("- Difficulty = Your team's fixture difficulty (2-5)")
print("- Opponent_Difficulty = Your opponent's fixture difficulty (2-5)")
print("- Difficulty_Difference = Your difficulty - Opponent's difficulty")
print()

# Individual variable analysis
print("=" * 50)
print("INDIVIDUAL VARIABLE ANALYSIS")
print("=" * 50)

# Simple regressions for each variable
for feature in feature_names:
    simple_model = LinearRegression()
    simple_model.fit(X[[feature]], y)
    simple_r2 = simple_model.score(X[[feature]], y)
    print(f"{feature}:")
    print(f"  R² Score: {simple_r2:.4f}")
    print(f"  Equation: y = {simple_model.intercept_:.3f} + {simple_model.coef_[0]:.3f} × {feature}")
    print()

# Predictions vs Actuals
print("=" * 50)
print("PREDICTIONS vs ACTUAL VALUES")
print("=" * 50)

results_df = pd.DataFrame({
    'Team_Index': range(len(y)),
    'Actual': y,
    'Predicted': y_pred,
    'Residual': y - y_pred,
    'Abs_Error': np.abs(y - y_pred)
})

results_df = results_df.sort_values('Abs_Error', ascending=False)
print("Top 10 Largest Prediction Errors:")
print(results_df.head(10).round(3))
print()

print("Model Summary Statistics:")
print(f"Best Predictions (MAE < 0.2): {len(results_df[results_df['Abs_Error'] < 0.2])} teams")
print(f"Worst Predictions (MAE > 0.5): {len(results_df[results_df['Abs_Error'] > 0.5])} teams")
print()

# Feature importance (absolute coefficients)
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Ranked by Absolute Coefficient Value:")
for _, row in feature_importance.iterrows():
    print(f"{row['Feature']}: {row['Coefficient']:.4f} (|{row['Abs_Coefficient']:.4f}|)")
print()

# Example predictions
print("=" * 50)
print("EXAMPLE PREDICTIONS")
print("=" * 50)

print("Example scenarios:")
examples = [
    [2, 3, -1, 1],  # Easy home fixture
    [4, 2, 2, 0],   # Hard away fixture but opponent easier
    [3, 3, 0, 1],   # Balanced home fixture
    [5, 3, 2, 0]    # Very hard away fixture
]

example_labels = [
    "Easy home fixture (Diff=2, Opp=3, Home)",
    "Hard away fixture, weak opponent (Diff=4, Opp=2, Away)",
    "Balanced home fixture (Diff=3, Opp=3, Home)",
    "Very hard away fixture (Diff=5, Opp=3, Away)"
]

for example, label in zip(examples, example_labels):
    prediction = model.predict([example])[0]
    print(f"{label}: {prediction:.2f} points")

print()
print("=" * 50)
print("MODEL INTERPRETATION")
print("=" * 50)

print("Key Insights:")
print(f"1. Each unit increase in your difficulty → {model.coef_[0]:.3f} point change")
print(f"2. Each unit increase in opponent difficulty → +{model.coef_[1]:.3f} point change")
print(f"3. Each unit increase in difficulty difference → {model.coef_[2]:.3f} point change")
print(f"4. Home advantage → +{model.coef_[3]:.3f} points")
print()

print("The model explains {:.1f}% of the variance in expected points.".format(r2 * 100))
