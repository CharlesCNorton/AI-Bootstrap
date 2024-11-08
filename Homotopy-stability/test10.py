import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import math

# Define a dataclass to hold simulation results
@dataclass
class SimulationResult:
    homotopy_type: str
    level: int
    epsilon: float
    perturbation: str
    value: float
    stable: bool

# Define the homotopy functions
def loop_space(a0: float, epsilon: float, n: int) -> float:
    """
    Computes the Loop Space L_n(a0, epsilon) as defined.
    """
    try:
        term1 = ((a0 + (a0 + epsilon)) / 2) ** (1 / n)
        term2 = np.cos(n * (a0 + epsilon))
        return term1 + term2
    except:
        return float('nan')

def product_type(a0_1: float, a0_2: float, epsilon: float, n: int) -> float:
    """
    Computes the Product Type P_n(a0^(1), a0^(2), epsilon) as defined.
    """
    try:
        term1 = (a0_1 + epsilon) ** (1 / n) + np.cos(n * (a0_1 + epsilon))
        term2 = (a0_2 - epsilon) ** (1 / n) + np.sin(n * (a0_2 - epsilon))
        return (term1 + term2) / 2
    except:
        return float('nan')

def fibration_type(a0_base: float, a0_fiber1: float, a0_fiber2: float, epsilon: float, n: int, cup_product1: float, cup_product2: float) -> float:
    """
    Computes the Fibration Type F_n(a0^(base), a0^(fiber1), a0^(fiber2), epsilon) with non-zero cup products.
    """
    try:
        term_base = (a0_base + epsilon) ** (1 / n) + np.cos(n * a0_base)
        term_fiber1 = ((a0_fiber1 + 0.5 * epsilon) ** (1 / (n + 1)) + np.sin(n * a0_fiber1) + cup_product1) / 2
        term_fiber2 = ((a0_fiber2 + 0.25 * epsilon) ** (1 / (n + 2)) + np.sin(n * a0_fiber2) + cup_product2) / 2
        return (term_base + term_fiber1 + term_fiber2) / 2
    except:
        return float('nan')

# Define the stability condition
def is_stable(value: float, delta: float = 2.0) -> bool:
    """
    Determines if a given value satisfies the stability condition |value| < delta.
    """
    return abs(value) < delta

# Simulation function
def run_simulation(
    homotopy_types: Dict[str, Callable],
    levels: range,
    epsilon_values_positive: list,
    epsilon_values_negative: list,
    a0_values: list,
    num_fiber: int = 2,  # Number of fiber components in fibration
    delta: float = 2.0,
    adaptive_scaling: bool = False
) -> pd.DataFrame:
    """
    Runs simulations across homotopy types, levels, and epsilon values.
    Returns a DataFrame with the results.
    """
    results = []

    for n in levels:
        for epsilon in epsilon_values_positive + epsilon_values_negative:
            # Determine if epsilon is positive or negative
            is_positive = epsilon > 0
            perturbation = 'Positive' if is_positive else 'Negative'
            # Adaptive scaling of epsilon if enabled
            if adaptive_scaling:
                # Example: Exponential scaling
                scaling_factor = np.exp(-n)
                scaled_epsilon = epsilon * scaling_factor
            else:
                scaled_epsilon = epsilon

            for a0 in a0_values:
                # Define cup products based on n and epsilon
                if 'Fibration Type' in homotopy_types:
                    # Example cup product values
                    cup_product1 = 0.1 * n * (1 if is_positive else -1)
                    cup_product2 = 0.05 * n * (1 if is_positive else -1)
                else:
                    cup_product1 = 0.0
                    cup_product2 = 0.0

                # Loop Space
                L = homotopy_types['Loop Space'](a0, scaled_epsilon, n)
                stable_L = is_stable(L, delta)
                results.append(SimulationResult('Loop Space', n, scaled_epsilon, perturbation, L, stable_L))

                # Product Type
                # For simplicity, assume a0^(1) = a0 and a0^(2) = a0 (can be randomized or varied)
                P = homotopy_types['Product Type'](a0, a0, scaled_epsilon, n)
                stable_P = is_stable(P, delta)
                results.append(SimulationResult('Product Type', n, scaled_epsilon, perturbation, P, stable_P))

                # Fibration Type
                F = homotopy_types['Fibration Type'](a0, a0, a0, scaled_epsilon, n, cup_product1, cup_product2)
                stable_F = is_stable(F, delta)
                results.append(SimulationResult('Fibration Type', n, scaled_epsilon, perturbation, F, stable_F))

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

# Statistical analysis function
def analyze_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the simulation results and returns statistical summaries.
    """
    summary = df.groupby(['homotopy_type', 'level', 'perturbation']).agg(
        Total_Evaluations = ('stable', 'count'),
        Stable = ('stable', 'sum'),
        Unstable = ('stable', lambda x: x.count() - x.sum()),
        Stability_Percentage = ('stable', lambda x: 100 * x.sum() / x.count()),
        Mean_Value = ('value', 'mean'),
        Std_Deviation = ('value', 'std')
    ).reset_index()

    return summary

# Main function to execute the simulation and analysis
def main():
    # Define simulation parameters
    homotopy_types = {
        'Loop Space': loop_space,
        'Product Type': product_type,
        'Fibration Type': fibration_type
    }

    # Define homotopy levels (e.g., 1 to 5)
    levels = range(1, 6)

    # Define epsilon perturbations separately for positive and negative
    epsilon_values_positive = [0.5, 0.25]
    epsilon_values_negative = [-0.5, -0.25]

    # Define a0 base points (e.g., integers from 1 to 10)
    a0_values = list(range(1, 11))

    # Run simulations without adaptive scaling
    print("Running simulations without adaptive scaling...")
    df = run_simulation(
        homotopy_types=homotopy_types,
        levels=levels,
        epsilon_values_positive=epsilon_values_positive,
        epsilon_values_negative=epsilon_values_negative,
        a0_values=a0_values,
        delta=2.0,
        adaptive_scaling=False
    )

    # Analyze results
    summary = analyze_results(df)
    print("\nSimulation Summary without Adaptive Scaling:")
    print(summary)

    # Run simulations with adaptive scaling
    print("\nRunning simulations with adaptive scaling...")
    df_adaptive = run_simulation(
        homotopy_types=homotopy_types,
        levels=levels,
        epsilon_values_positive=epsilon_values_positive,
        epsilon_values_negative=epsilon_values_negative,
        a0_values=a0_values,
        delta=2.0,
        adaptive_scaling=True
    )

    # Analyze adaptive scaling results
    summary_adaptive = analyze_results(df_adaptive)
    print("\nSimulation Summary with Adaptive Scaling:")
    print(summary_adaptive)

    # Optionally, save results to CSV
    df.to_csv('simulation_results.csv', index=False)
    summary.to_csv('simulation_summary.csv', index=False)
    df_adaptive.to_csv('simulation_results_adaptive.csv', index=False)
    summary_adaptive.to_csv('simulation_summary_adaptive.csv', index=False)

    print("\nSimulation complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()
