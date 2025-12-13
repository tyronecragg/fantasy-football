import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class FootballMultiPredictor:
    def __init__(self, degree=1):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = [
            'Score 1+', 'Assist', 'Yellow Card', 'Clean Sheet',
            'Concede 2+ Goals', 'Concede 4+ Goals', '3+ Saves', '6+ Saves'
        ]

    def prepare_data(self, df):
        """Prepare features and targets from the dataframe"""
        # Input features: Win, Opponent Win, Diff, Venue
        feature_columns = ['Win', 'Opponent Win', 'Diff', 'Venue']

        # Check if columns exist, if not try alternative names
        if 'Win' not in df.columns:
            # Try alternative column names from the document
            alt_mapping = {
                'Win': 'Win',
                'Opponent Win': 'Opponent Win',
                'Diff': 'Diff',
                'Venue': 'Venue'
            }

            # Find actual column names (case insensitive)
            actual_columns = []
            for expected in feature_columns:
                found = False
                for col in df.columns:
                    if col.lower().replace(' ', '').replace('_', '') == expected.lower().replace(' ', '').replace('_',
                                                                                                                  ''):
                        actual_columns.append(col)
                        found = True
                        break
                if not found:
                    actual_columns.append(expected)  # Keep original if not found
            feature_columns = actual_columns

        # Encode venue (H=1, A=0)
        X = df[feature_columns].copy()
        X['Venue_Encoded'] = self.label_encoder.fit_transform(X['Venue'].astype(str))

        # Drop original venue column and use encoded version
        X_final = X[['Win', 'Opponent Win', 'Diff', 'Venue_Encoded']].copy()

        # Target columns - handle missing values by using available columns
        available_targets = []
        target_columns = []

        for target in self.target_names:
            # Try to find the target column (handling variations in naming)
            found_col = None
            for col in df.columns:
                if target.lower().replace(' ', '').replace('+', '') in col.lower().replace(' ', '').replace('+', ''):
                    found_col = col
                    break

            if found_col is not None:
                available_targets.append(target)
                target_columns.append(found_col)

        print(f"Found {len(available_targets)} target columns: {available_targets}")

        # Get target data, handling missing values
        if target_columns:
            y = df[target_columns].copy()
            # Replace '#N/A' strings and convert to float
            y = y.replace('#N/A', np.nan).astype(float)
        else:
            # If no target columns found, create dummy data for demonstration
            print("Warning: No target columns found. Creating dummy targets for demonstration.")
            y = pd.DataFrame(np.random.rand(len(X_final), len(self.target_names)),
                             columns=self.target_names)
            available_targets = self.target_names

        return X_final, y, available_targets

    def fit(self, X, y, available_targets):
        """Train models for each target"""
        # Create polynomial features
        X_poly = self.poly_features.fit_transform(X)
        self.feature_names = self.poly_features.get_feature_names_out(X.columns)

        # Train a model for each target
        for i, target in enumerate(available_targets):
            if i < y.shape[1]:  # Check if target exists
                # Get non-null values for this target
                mask = ~y.iloc[:, i].isna()
                if mask.sum() > 10:  # Only train if we have enough data
                    X_target = X_poly[mask]
                    y_target = y.iloc[:, i][mask]

                    model = LinearRegression()
                    model.fit(X_target, y_target)
                    self.models[target] = {
                        'model': model,
                        'mask_count': mask.sum()
                    }
                    print(f"Trained model for {target} with {mask.sum()} samples")
                else:
                    print(f"Insufficient data for {target} ({mask.sum()} samples)")

    def predict(self, X):
        """Make predictions for all targets"""
        X_poly = self.poly_features.transform(X)
        predictions = {}

        for target, model_info in self.models.items():
            model = model_info['model']
            pred = model.predict(X_poly)
            predictions[target] = pred

        return predictions

    def get_formulas(self):
        """Generate polynomial formulas for each target"""
        formulas = {}

        for target, model_info in self.models.items():
            model = model_info['model']
            coefficients = model.coef_
            intercept = model.intercept_

            # Create readable feature names mapping
            feature_mapping = {
                'Win': 'W',
                'Opponent Win': 'OW',
                'Diff': 'D',
                'Venue_Encoded': 'V'
            }

            formula = f"{target} = {intercept:.6f}"

            for coef, feature_name in zip(coefficients, self.feature_names):
                if abs(coef) > 1e-10:  # Only include non-zero coefficients
                    # Clean up the feature name for readability
                    readable_feature = feature_name
                    for original, short in feature_mapping.items():
                        readable_feature = readable_feature.replace(original, short)

                    # Fix spaces in interaction terms
                    readable_feature = readable_feature.replace(' ', ' * ')

                    sign = " + " if coef >= 0 else " - "
                    formula += f"{sign}{abs(coef):.6f} * {readable_feature}"

            formulas[target] = formula

        return formulas

    def evaluate_models(self, X, y, available_targets):
        """Evaluate model performance"""
        X_poly = self.poly_features.transform(X)
        results = {}

        for i, target in enumerate(available_targets):
            if target in self.models and i < y.shape[1]:
                mask = ~y.iloc[:, i].isna()
                if mask.sum() > 0:
                    X_target = X_poly[mask]
                    y_true = y.iloc[:, i][mask]
                    y_pred = self.models[target]['model'].predict(X_target)

                    results[target] = {
                        'r2_score': r2_score(y_true, y_pred),
                        'mse': mean_squared_error(y_true, y_pred),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'samples': len(y_true)
                    }

        return results


def predict_football_performance(win_prob, opponent_win_prob, diff, venue):
    """
    Predict football performance metrics

    Parameters:
    - win_prob: Team's win probability (0.0-1.0)
    - opponent_win_prob: Opponent's win probability (0.0-1.0)
    - diff: Difference between team strengths (-1.0 to 1.0)
    - venue: 'H' for home, 'A' for away

    Returns:
    - Dictionary with predictions for each target
    """
    # This would be set after training the model
    if 'trained_predictor' in globals():
        venue_encoded = 1 if venue == 'H' else 0
        input_data = pd.DataFrame({
            'Win': [win_prob],
            'Opponent Win': [opponent_win_prob],
            'Diff': [diff],
            'Venue_Encoded': [venue_encoded]
        })

        predictions = trained_predictor.predict(input_data)
        return predictions
    else:
        print("Model not trained yet. Please train the model first.")
        return {}


# Main execution
def main():

    df = pd.read_csv('football_data.csv')
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Initialize and train the predictor
    global trained_predictor
    trained_predictor = FootballMultiPredictor(degree=1)

    # Prepare data
    X, y, available_targets = trained_predictor.prepare_data(df)
    print(f"\nPrepared features shape: {X.shape}")
    print(f"Prepared targets shape: {y.shape}")

    # Train models
    print(f"\nTraining models for {len(available_targets)} targets...")
    trained_predictor.fit(X, y, available_targets)

    # Evaluate models
    print(f"\nModel Performance:")
    print("=" * 60)
    results = trained_predictor.evaluate_models(X, y, available_targets)

    for target, metrics in results.items():
        print(f"\n{target}:")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Samples: {metrics['samples']}")

    # Generate formulas
    print(f"\n" + "=" * 60)
    print("POLYNOMIAL FORMULAS:")
    print("=" * 60)

    formulas = trained_predictor.get_formulas()
    for target, formula in formulas.items():
        print(f"\n{target}:")
        print(formula)

    # Example predictions
    print(f"\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS:")
    print("=" * 60)

    examples = [
        (0.7, 0.3, 0.4, 'H'),  # Strong team at home
        (0.3, 0.7, -0.4, 'A'),  # Weak team away
        (0.5, 0.5, 0.0, 'H'),  # Even match at home
    ]

    for i, (win_prob, opp_win_prob, diff, venue) in enumerate(examples, 1):
        print(f"\nExample {i}: Win={win_prob}, Opp Win={opp_win_prob}, Diff={diff}, Venue={venue}")
        predictions = predict_football_performance(win_prob, opp_win_prob, diff, venue)

        for target, pred_value in predictions.items():
            print(f"  {target}: {pred_value[0]:.3f}")

    # Visualization
    if len(results) > 0:
        plt.figure(figsize=(12, 8))

        # Plot R² scores
        plt.subplot(2, 2, 1)
        targets = list(results.keys())
        r2_scores = [results[t]['r2_score'] for t in targets]
        plt.bar(range(len(targets)), r2_scores)
        plt.xticks(range(len(targets)), targets, rotation=45)
        plt.ylabel('R² Score')
        plt.title('Model Performance (R² Score)')
        plt.tight_layout()

        # Plot sample counts
        plt.subplot(2, 2, 2)
        sample_counts = [results[t]['samples'] for t in targets]
        plt.bar(range(len(targets)), sample_counts)
        plt.xticks(range(len(targets)), targets, rotation=45)
        plt.ylabel('Number of Samples')
        plt.title('Training Data Availability')
        plt.tight_layout()

        # Feature importance (if degree=1)
        if trained_predictor.degree == 1 and len(trained_predictor.models) > 0:
            plt.subplot(2, 2, 3)
            first_target = list(trained_predictor.models.keys())[0]
            coeffs = trained_predictor.models[first_target]['model'].coef_
            feature_names = ['Win', 'Opp Win', 'Diff', 'Venue']
            plt.bar(feature_names, np.abs(coeffs))
            plt.ylabel('Absolute Coefficient')
            plt.title(f'Feature Importance ({first_target})')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    print(f"\n" + "=" * 60)
    print("USAGE INSTRUCTIONS:")
    print("=" * 60)
    print("Use predict_football_performance(win_prob, opponent_win_prob, diff, venue)")
    print("- win_prob: Team's win probability (0.0-1.0)")
    print("- opponent_win_prob: Opponent's win probability (0.0-1.0)")
    print("- diff: Difference between probabilities (-1.0 to 1.0)")
    print("- venue: 'H' for home, 'A' for away")

    return trained_predictor, results


# Run the main function
if __name__ == "__main__":
    predictor, results = main()
