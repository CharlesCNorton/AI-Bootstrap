import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict
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

# Function to check if a number is prime
def is_prime(num: int) -> bool:
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

# Function to generate a0 values based on number-theoretic constraints
def generate_a0_values() -> list:
    """
    Generates a0 values that are prime numbers and satisfy specific modular congruences.
    """
    primes = [num for num in range(2, 50) if is_prime(num)]
    # Example: a0 ≡ 1 mod 3 or a0 ≡ 0 mod 5
    filtered_a0 = [a0 for a0 in primes if (a0 % 3 == 1) or (a0 % 5 == 0)]
    return filtered_a0

# Function to generate epsilon values based on rational fractions with small denominators
def generate_epsilon_values() -> list:
    """
    Generates epsilon values that are rational fractions with small denominators.
    """
    denominators = [2, 3, 4, 5]
    fractions = []
    for d in denominators:
        fractions.extend([1/d, -1/d])
    return fractions

# Simulation function for n=1 with number-theoretic constraints
def run_simulation_n1(
    homotopy_types: Dict[str, Callable],
    a0_values: list,
    epsilon_values: list,
    delta: float = 2.0
) -> pd.DataFrame:
    """
    Runs simulations specifically for n=1 with number-theoretic constraints on a0 and epsilon.
    Returns a DataFrame with the results.
    """
    results = []
    n = 1  # Fixed homotopy level

    for epsilon in epsilon_values:
        # Determine if epsilon is positive or negative
        is_positive = epsilon > 0
        perturbation = 'Positive' if is_positive else 'Negative'

        for a0 in a0_values:
            # Define cup products based on n and epsilon
            # For n=1, cup products can be set based on specific number-theoretic properties
            cup_product1 = 0.1 * n * (1 if is_positive else -1)
            cup_product2 = 0.05 * n * (1 if is_positive else -1)

            # Loop Space
            L = homotopy_types['Loop Space'](a0, epsilon, n)
            stable_L = is_stable(L, delta)
            results.append(SimulationResult('Loop Space', n, epsilon, perturbation, L, stable_L))

            # Product Type
            P = homotopy_types['Product Type'](a0, a0, epsilon, n)
            stable_P = is_stable(P, delta)
            results.append(SimulationResult('Product Type', n, epsilon, perturbation, P, stable_P))

            # Fibration Type
            F = homotopy_types['Fibration Type'](a0, a0, a0, epsilon, n, cup_product1, cup_product2)
            stable_F = is_stable(F, delta)
            results.append(SimulationResult('Fibration Type', n, epsilon, perturbation, F, stable_F))

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

# Statistical analysis function
def analyze_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the simulation results and returns statistical summaries.
    """
    summary = df.groupby(['homotopy_type', 'perturbation']).agg(
        Total_Evaluations = ('stable', 'count'),
        Stable = ('stable', 'sum'),
        Unstable = ('stable', lambda x: x.count() - x.sum()),
        Stability_Percentage = ('stable', lambda x: 100 * x.sum() / x.count()),
        Mean_Value = ('value', 'mean'),
        Std_Deviation = ('value', 'std')
    ).reset_index()

    return summary

# Main function to execute the simulation and analysis for n=1
def main():
    # Define homotopy types
    homotopy_types = {
        'Loop Space': loop_space,
        'Product Type': product_type,
        'Fibration Type': fibration_type
    }

    # Generate a0 values based on number-theoretic constraints
    a0_values = generate_a0_values()

    # Generate epsilon values based on rational fractions with small denominators
    epsilon_values = generate_epsilon_values()

    # Run simulation for n=1
    df_n1 = run_simulation_n1(
        homotopy_types=homotopy_types,
        a0_values=a0_values,
        epsilon_values=epsilon_values,
        delta=2.0
    )

    # Analyze results
    summary_n1 = analyze_results(df_n1)
    print("\nSimulation Summary for n=1 with Number-Theoretic Constraints:")
    print(summary_n1)

    # Optionally, save results to CSV
    df_n1.to_csv('simulation_n1_results.csv', index=False)
    summary_n1.to_csv('simulation_n1_summary.csv', index=False)

    print("\nSimulation for n=1 complete. Results saved to 'simulation_n1_results.csv' and 'simulation_n1_summary.csv'.")

if __name__ == "__main__":
    main()
