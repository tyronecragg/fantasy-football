import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns


def load_and_clean_data(file_path):
    """
    Load and clean the XP and Bonus data
    """
    # Read the data
    df = pd.read_csv(file_path)

    # Clean the bonus column (remove % sign and convert to decimal)
    df['Bonus'] = df['Bonus'].str.rstrip('%').astype(float) / 100

    # Remove rows where XP is 0 (optional - you can keep them if needed)
    df_clean = df[df['XP'] > 0].copy()

    print(f"Original data points: {len(df)}")
    print(f"Data points after removing XP=0: {len(df_clean)}")
    print(f"XP range: {df_clean['XP'].min():.2f} to {df_clean['XP'].max():.2f}")
    print(
        f"Bonus range: {df_clean['Bonus'].min():.1%} to {df_clean['Bonus'].max():.1%}")

    return df_clean


def create_linear_model(df):
    """
    Create and train a linear regression model
    """
    X = df[['XP']]  # Features (needs to be 2D for sklearn)
    y = df['Bonus']  # Target variable

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print("\n" + "=" * 50)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 50)
    print(f"Formula: Bonus = {model.intercept_:.6f} + {model.coef_[0]:.6f} * XP")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean Squared Error: {mse:.8f}")

    return model, y_pred, r2, rmse


def predict_bonus_probability(model, xp_value):
    """
    Predict bonus for a given XP value
    """
    prediction = model.predict([[xp_value]])[0]
    return max(0, min(1, prediction))  # Ensure result is between 0 and 1


def plot_results(df, model, y_pred):
    """
    Create visualizations of the data and model
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot with regression line
    plt.subplot(1, 3, 1)
    plt.scatter(df['XP'], df['Bonus'], alpha=0.6, color='blue', s=20)
    plt.plot(df['XP'], y_pred, color='red', linewidth=2, label='Linear Regression')
    plt.xlabel('XP')
    plt.ylabel('Bonus')
    plt.title('XP vs Bonus')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals plot
    plt.subplot(1, 3, 2)
    residuals = df['Bonus'] - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color='green', s=20)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Bonus')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)

    # Plot 3: Distribution of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the complete analysis
    """
    # Load and clean data
    df = load_and_clean_data('bonus_points.csv')  # Replace with your file path

    # Create linear model
    model, y_pred, r2, rmse = create_linear_model(df)

    # Create visualizations
    plot_results(df, model, y_pred)

    # Example predictions
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)
    test_xp_values = [1.0, 2.5, 4.0, 5.5, 7.0]

    for xp in test_xp_values:
        predicted_prob = predict_bonus_probability(model, xp)
        print(f"XP = {xp:4.1f} → Predicted Bonus = {predicted_prob:.1%}")

    # Function to use the model
    print("\n" + "=" * 50)
    print("HOW TO USE THE MODEL")
    print("=" * 50)
    print("def predict_bonus_probability(xp):")
    print(f"    return max(0, min(1, {model.intercept_:.6f} + {model.coef_[0]:.6f} * xp))")
    print("\n# Example usage:")
    print("# bonus_prob = predict_bonus_probability(3.5)")
    print("# print(f'Bonus: {bonus_prob:.1%}')")

    return model, df


# Run the analysis
if __name__ == "__main__":
    model, data = main()

    # Additional statistics
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(data[['XP', 'Bonus']].describe())
