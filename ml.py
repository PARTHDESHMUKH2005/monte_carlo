"""


This file contains all core logic for the Monte Carlo + ML surrogate project.

Features:
1. Monte Carlo simulation of capital with cash inflows/outflows.
2. Drawdown calculation to track worst-case losses.
3. Scenario-based simulation support (Normal, High Volatility, High Burn).
4. ML surrogate (Random Forest) to predict VaR 95% instantly.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#logic
def run_simulation(initial_capital, daily_inflow, daily_outflow, volatility, n_steps=252, n_simulations=500):
    """
    Simulates capital paths over time using Monte Carlo method.

    Parameters:
    - initial_capital: starting capital
    - daily_inflow: cash coming in per day
    - daily_outflow: cash spent per day
    - volatility: standard deviation of daily returns
    - n_steps: number of time steps (e.g., 252 trading days)
    - n_simulations: number of Monte Carlo paths

    Returns:
    - all_paths: array of simulated capital paths (n_simulations x n_steps)
    - VaR_95: 5th percentile of final capital (risk measure)
    """
    all_paths = np.zeros((n_simulations, n_steps))

    for i in range(n_simulations):
        # Generate random daily returns
        daily_returns = np.random.normal(0.0005, volatility, n_steps)

        # Initialize capital array for this simulation
        capital = np.zeros(n_steps)
        capital[0] = initial_capital + daily_inflow - daily_outflow

        # Simulate capital over time
        for t in range(1, n_steps):
            capital[t] = capital[t-1] * (1 + daily_returns[t]) + daily_inflow - daily_outflow

        all_paths[i] = capital

    # Calculate Value at Risk 95% (VaR)
    VaR_95 = np.percentile(all_paths[:, -1], 5)

    return all_paths, VaR_95


# Function 2: Calculate drawdown for each path
def calculate_drawdown(all_paths):
    """
    Calculates max drawdown for each simulated path.

    Parameters:
    - all_paths: array of Monte Carlo capital paths

    Returns:
    - drawdowns: list of max drawdowns for each path
    """
    drawdowns = []

    for path in all_paths:
        peak = path[0]
        max_dd = 0
        for value in path:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        drawdowns.append(max_dd)

    return drawdowns


# Function 3: Apply scenario modifications
def apply_scenario(initial_capital, daily_inflow, daily_outflow, volatility, scenario):
    """
    Modifies parameters based on selected scenario.

    Parameters:
    - initial_capital: starting capital
    - daily_inflow: cash coming in per day
    - daily_outflow: cash spent per day
    - volatility: standard deviation of daily returns
    - scenario: str, one of ["Normal", "High Volatility", "High Burn"]

    Returns:
    - modified parameters (tuple)
    """
    if scenario == "High Volatility":
        volatility *= 2  # double volatility
    elif scenario == "High Burn":
        daily_outflow *= 1.5  # increase daily outflow by 50%

    return initial_capital, daily_inflow, daily_outflow, volatility


# Function 4: Train Random Forest surrogate to predict VaR
def train_ml_surrogate(initial_capital, daily_inflow, daily_outflow, volatility, n_simulations=500):
    """
    Trains a Random Forest model to approximate Monte Carlo VaR 95%.

    Parameters:
    - initial_capital: starting capital
    - daily_inflow: cash coming in per day
    - daily_outflow: cash spent per day
    - volatility: standard deviation of daily returns
    - n_simulations: number of samples for ML training

    Returns:
    - rf_model: trained Random Forest model
    - mse: mean squared error on test data
    """
    # Step 1: Generate varied training samples
    np.random.seed(42)
    
    # Create feature variations around the input parameters
    features = []
    targets = []
    
    for _ in range(n_simulations):
        # Add some variation to create diverse training samples
        ic_sample = initial_capital * np.random.uniform(0.8, 1.2)
        inflow_sample = daily_inflow * np.random.uniform(0.8, 1.2)
        outflow_sample = daily_outflow * np.random.uniform(0.8, 1.2)
        vol_sample = volatility * np.random.uniform(0.5, 1.5)
        
        features.append([ic_sample, inflow_sample, outflow_sample, vol_sample])
        
        # Generate target VaR for this sample
        _, var = run_simulation(ic_sample, inflow_sample, outflow_sample, vol_sample, n_simulations=100)
        targets.append(var)
    
    features = np.array(features)
    targets = np.array(targets)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Step 3: Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Step 4: Evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return rf_model, mse


# Function 5: Predict VaR using trained ML model
def predict_var_ml(rf_model, initial_capital, daily_inflow, daily_outflow, volatility):
    """
    Uses trained Random Forest to predict VaR instantly.

    Parameters:
    - rf_model: trained Random Forest model
    - initial_capital: starting capital
    - daily_inflow: cash coming in per day
    - daily_outflow: cash spent per day
    - volatility: standard deviation of daily returns

    Returns:
    - predicted VaR 95%
    """
    features = np.array([[initial_capital, daily_inflow, daily_outflow, volatility]])
    return rf_model.predict(features)[0]


# Function 6: Calculate additional risk metrics
def calculate_risk_metrics(all_paths):
    """
    Calculates comprehensive risk metrics from simulation paths.

    Parameters:
    - all_paths: array of simulated capital paths

    Returns:
    - dict: dictionary containing various risk metrics
    """
    final_values = all_paths[:, -1]
    
    metrics = {
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'std': np.std(final_values),
        'var_95': np.percentile(final_values, 5),
        'var_99': np.percentile(final_values, 1),
        'best_case': np.max(final_values),
        'worst_case': np.min(final_values),
        'prob_loss': np.mean(final_values < all_paths[:, 0]) * 100
    }
    
    return metrics


# Function 7: Identify critical paths
def identify_critical_paths(all_paths, threshold_percentile=5):
    """
    Identifies simulation paths that fall below a critical threshold.

    Parameters:
    - all_paths: array of simulated capital paths
    - threshold_percentile: percentile to use as threshold

    Returns:
    - critical_indices: indices of critical paths
    - threshold_value: the VaR threshold value
    """
    final_values = all_paths[:, -1]
    threshold_value = np.percentile(final_values, threshold_percentile)
    critical_indices = np.where(final_values <= threshold_value)[0]
    
    return critical_indices, threshold_value