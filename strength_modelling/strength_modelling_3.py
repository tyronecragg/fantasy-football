import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def load_and_process_data(file_path):
    """
    Load and process the football data from CSV file
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean column names (remove any extra spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Process the data
    processed_df = df.copy()
    
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            # Handle percentage columns
            if processed_df[col].astype(str).str.contains('%', na=False).any():
                # Remove % sign and convert to float
                processed_df[col] = processed_df[col].astype(str).str.replace('%', '')
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Handle #N/A values
            processed_df[col] = processed_df[col].replace('#N/A', np.nan)
            processed_df[col] = processed_df[col].replace('N/A', np.nan)
            
            # Try to convert to numeric if not venue column
            if col != 'F1 Venue':
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    return processed_df


def create_linear_formulas(file_path='football_data.csv'):
    """
    Creates linear formulas for all output variables using Difficulty, Opponent, Diff, and Venue as inputs.
    
    Parameters:
    - file_path: Path to the CSV file containing the football data
    """
    
    # Load and process data
    df = load_and_process_data(file_path)
    
    # Define input features
    input_features = ['Win', 'Opponent Win', 'Diff', 'Venue']
    
    # Check if all input features exist
    missing_features = [feat for feat in input_features if feat not in df.columns]
    if missing_features:
        print(f"Error: Missing input features: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Output variables (all other columns)
    output_variables = [col for col in df.columns if col not in input_features]
    
    # Prepare input data
    X = df[input_features].copy()

    # Handle venue encoding (H=1, A=0)
    if X['Venue'].dtype == 'object':
        X['Venue'] = X['Venue'].map({'H': 1, 'A': 0})
    
    print("\n" + "="*80)
    print("LINEAR FORMULAS FOR FOOTBALL PREDICTION MODEL")
    print("="*80)
    print(f"Input Features: Win (%), Opponent Win (%), Diff, Venue (H=1, A=0)")
    print(f"Total variables to model: {len(output_variables)}")
    print("\nFormula format: Y = a + b*Win + c*OpponentWin + d*Diff + e*Venue")
    print("="*80)
    
    formulas = {}
    successful_models = 0
    
    for target_var in output_variables:
        y = df[target_var].copy()
        
        # Create mask for valid data (no NaN values)
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        
        if valid_mask.sum() < 5:  # Need at least 5 data points
            print(f"\n{target_var}: Insufficient data ({valid_mask.sum()} valid points) - SKIPPED")
            continue
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_valid, y_valid)
            
            # Extract coefficients
            intercept = model.intercept_
            coefficients = model.coef_
            
            # Calculate R-squared
            r2_score = model.score(X_valid, y_valid)

            # Create formula string
            formula_parts = [f"{intercept:.4f}"]
            feature_names = ['Win', 'Opponent Win', 'Diff', 'Venue']
            
            for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
                if coef >= 0:
                    formula_parts.append(f" + {coef:.6f}*{feature}")
                else:
                    formula_parts.append(f" - {abs(coef):.6f}*{feature}")
            
            formula = "".join(formula_parts)

            # Store formula
            formulas[target_var] = {
                'formula': formula,
                'coefficients': {
                    'intercept': intercept,
                    'win': coefficients[0],
                    'opponent_win': coefficients[1],
                    'diff': coefficients[2],
                    'venue': coefficients[3]
                },
                'r2_score': r2_score,
                'n_samples': valid_mask.sum(),
                'mean_y': y_valid.mean(),
                'std_y': y_valid.std()
            }
            
            print(f"\n{target_var}:")
            print(f"  {target_var} = {formula}")
            print(f"  R² = {r2_score:.4f}, n = {valid_mask.sum()}, mean = {y_valid.mean():.2f}")
            
            successful_models += 1
            
        except Exception as e:
            print(f"\n{target_var}: Error fitting model - {str(e)}")
            continue
    
    print(f"\n" + "="*80)
    print(f"SUMMARY: Successfully created {successful_models} linear models out of {len(output_variables)} variables")
    print("="*80)
    
    return formulas


def predict_outcomes(formulas, win, opponent_win, diff, venue):
    
    if not formulas:
        print("No formulas available for prediction")
        return None
    
    venue_encoded = 1 if venue.upper() == 'H' else 0

    print(f"\nPREDICTIONS")
    print("=" * 60)
    print(f"Inputs: Difficulty={win}%, OpponentWin={opponent_win}%, Diff={diff}, Venue={venue}")
    print("\nPredicted Outcomes:")
    print("-" * 60)

    predictions = {}
    
    for var_name, formula_data in formulas.items():
        coef = formula_data['coefficients']
        prediction = (coef['intercept'] + 
                     coef['win'] * win +
                     coef['opponent_win'] * opponent_win +
                     coef['diff'] * diff +
                     coef['venue'] * venue_encoded)
        
        predictions[var_name] = prediction
        r2 = formula_data['r2_score']
        print(f"{var_name:25}: {prediction:8.2f} (R² = {r2:.3f})")
    
    return predictions


def main():
    """
    Main function to demonstrate usage
    """
    print("Football Data Linear Regression Analysis")
    print("="*50)
    
    # Create formulas from the CSV file
    formulas = create_linear_formulas('football_data.csv')
    
    if formulas:
        # Example predictions
        print("\n" + "="*60)
        print("EXAMPLE PREDICTIONS")
        print("="*60)

        # Example 1: High difficulty home game
        predict_outcomes(formulas, win=.85, opponent_win=.15, diff=.7, venue='H')
        
        # Example 2: Medium difficulty away game
        predict_outcomes(formulas, win=.45, opponent_win=.25, diff=.2, venue='A')
        
        # Example 3: Low difficulty home game
        predict_outcomes(formulas, win=.1, opponent_win=.05, diff=.05, venue='H')
        
        print("\n" + "="*60)
        print("Analysis complete! Check 'football_formulas.txt' for all formulas.")
        print("="*60)
    
    return formulas

# Usage examples:
if __name__ == "__main__":
    # Run the main analysis
    formulas = main()
