import math
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Callable, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Define a dataclass to store the results of each test
@dataclass
class AdvancedStabilityResult:
    homotopy_level: int
    homotopy_type: str
    perturbation_type: str
    perturbation_value: float
    total_evaluations: int
    stable_count: int
    unstable_count: int
    stability_percentage: float
    mean_value: float
    std_deviation: float
    confidence_interval: Tuple[float, float]

# Define the perturbation scaling functions for adaptive scaling
def adaptive_scaling(n: int) -> float:
    return 1.0 / (1 + n)

def adaptive_scaling_negative(n: int) -> float:
    return -1.0 / (1 + n)

# Define the symbolic variables
a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3')

# Define advanced homotopy functions using SymPy for symbolic computation

def advanced_loop_space(n: int, a0_val: float, perturbation: float) -> float:
    """
    Defines the Loop Space L_n(a0, perturbation) with higher complexity.
    Incorporates iterated loop operations and non-linear perturbations.
    """
    # Example: L_n(a0, epsilon) = ((a0 + P(a0, epsilon)) / (1 + a0))**(1/n) + cos(n * (a0 + epsilon)) * sin(a0)
    # Introduce non-linear terms for higher complexity
    try:
        P = a0_val + perturbation
        term1 = ((a0_val + P) / (1 + a0_val))**(1 / n)
        term2 = math.cos(n * (a0_val + perturbation)) * math.sin(a0_val)
        term3 = math.exp(-a0_val) * math.tanh(P)
        result = term1 + term2 + term3
    except ZeroDivisionError:
        result = float('inf')
    return result

def advanced_product_type(n: int, a0_val: float, a1_val: float, a2_val: float, perturbation: float) -> float:
    """
    Defines the Product Type P_n(a0, a1, a2, perturbation) with higher complexity.
    Incorporates multi-dimensional interactions and non-linear perturbations.
    """
    try:
        P = a0_val + perturbation
        Q = a1_val - perturbation
        R = a2_val + 0.5 * perturbation
        term1 = (P * Q)**(1 / n)
        term2 = math.sin(n * P) * math.cos(n * Q)
        term3 = (R**2) / (1 + R)
        result = (term1 + term2 + term3) / 3
    except ZeroDivisionError:
        result = float('inf')
    return result

def advanced_fibration_type(n: int, a0_base: float, a1_fiber1: float, a2_fiber2: float, a3_fiber3: float, perturbation: float) -> float:
    """
    Defines the Fibration Type F_n(a0_base, a1_fiber1, a2_fiber2, a3_fiber3, perturbation) with higher complexity.
    Incorporates nested fibrations and cohomological interactions.
    """
    try:
        P = a0_base + perturbation
        Q = a1_fiber1 + 0.5 * perturbation
        R = a2_fiber2 + 0.25 * perturbation
        S = a3_fiber3 + 0.1 * perturbation
        term_base = (P**2 + math.cos(n * P)) / (1 + P)
        term_fiber1 = math.sin(n * Q) + math.log(1 + Q)
        term_fiber2 = math.exp(-R) * math.tanh(R)
        term_fiber3 = S**(1 / (n + 1)) + math.sqrt(abs(S))
        # Nested fibrations
        nested_fibration = (term_fiber1 * term_fiber2) / (1 + term_fiber3)
        result = (term_base + nested_fibration) / 2
    except (ZeroDivisionError, ValueError):
        result = float('inf')
    return result

# Define mapping from homotopy type to function
def get_homotopy_function(homotopy_type: str) -> Callable:
    if homotopy_type == "Loop Space":
        return advanced_loop_space
    elif homotopy_type == "Product Type":
        return advanced_product_type
    elif homotopy_type == "Fibration Type":
        return advanced_fibration_type
    else:
        raise ValueError(f"Unknown homotopy type: {homotopy_type}")

# Define the homotopy group simulation function
def simulate_homotopy(homotopy_level: int, homotopy_type: str, perturbation: Tuple[str, float], a_values: List[float]) -> AdvancedStabilityResult:
    """
    Simulates a single homotopy test.

    Parameters:
        homotopy_level (int): The homotopy level n.
        homotopy_type (str): The type of homotopy structure.
        perturbation (Tuple[str, float]): The perturbation type and value.
        a_values (List[float]): The list of a0, a1, a2, ... values.

    Returns:
        AdvancedStabilityResult: The result of the stability test.
    """
    perturbation_type, epsilon = perturbation
    # Assign a_values based on homotopy type
    if homotopy_type == "Loop Space":
        a0_val = a_values[0]
        output = get_homotopy_function(homotopy_type)(homotopy_level, a0_val, epsilon)
    elif homotopy_type == "Product Type":
        a0_val, a1_val, a2_val = a_values[:3]
        output = get_homotopy_function(homotopy_type)(homotopy_level, a0_val, a1_val, a2_val, epsilon)
    elif homotopy_type == "Fibration Type":
        a0_base, a1_fiber1, a2_fiber2, a3_fiber3 = a_values[:4]
        output = get_homotopy_function(homotopy_type)(homotopy_level, a0_base, a1_fiber1, a2_fiber2, a3_fiber3, epsilon)
    else:
        raise ValueError(f"Unknown homotopy type: {homotopy_type}")

    # Define stability condition
    delta = 2.0  # Threshold for stability
    is_stable = abs(output) < delta

    return is_stable, output

# Function to perform advanced stability tests with multiprocessing
def perform_advanced_stability_tests(
    homotopy_levels: List[int],
    homotopy_types: List[str],
    perturbations: List[Tuple[str, float]],
    evaluations_per_test: int = 100000,
    a0_range: Tuple[float, float] = (0, 2 * math.pi),
    a1_range: Tuple[float, float] = (0, 2 * math.pi),
    a2_range: Tuple[float, float] = (0, 2 * math.pi),
    a3_range: Tuple[float, float] = (0, 2 * math.pi)
) -> List[AdvancedStabilityResult]:
    """
    Performs advanced stability tests across multiple homotopy levels and types.

    Parameters:
        homotopy_levels (List[int]): List of homotopy levels to test.
        homotopy_types (List[str]): List of homotopy types to test.
        perturbations (List[Tuple[str, float]]): List of perturbation types and values.
        evaluations_per_test (int): Number of evaluations per test.
        a*_range (Tuple[float, float]): Ranges for a0, a1, a2, a3 values.

    Returns:
        List[AdvancedStabilityResult]: List of results from all tests.
    """
    results = []
    pool = Pool(processes=cpu_count())

    for n in homotopy_levels:
        for homotopy_type in homotopy_types:
            for perturbation in perturbations:
                # Define the perturbation type and value
                perturbation_type, epsilon = perturbation
                # Generate random a_values based on homotopy type
                if homotopy_type == "Loop Space":
                    a0_vals = np.random.uniform(a0_range[0], a0_range[1], evaluations_per_test)
                    # For multiprocessing, we'll split the evaluations into chunks
                    chunk_size = 1000
                    chunks = [a0_vals[i:i + chunk_size] for i in range(0, evaluations_per_test, chunk_size)]
                    func = partial(simulate_homotopy, n, homotopy_type, perturbation)
                    # Prepare input data
                    input_data = [(chunk[0],) for chunk in chunks]  # a0_val
                elif homotopy_type == "Product Type":
                    a0_vals = np.random.uniform(a0_range[0], a0_range[1], evaluations_per_test)
                    a1_vals = np.random.uniform(a1_range[0], a1_range[1], evaluations_per_test)
                    a2_vals = np.random.uniform(a2_range[0], a2_range[1], evaluations_per_test)
                    chunk_size = 1000
                    chunks = list(zip(a0_vals, a1_vals, a2_vals))
                    func = partial(simulate_homotopy, n, homotopy_type, perturbation)
                    input_data = [chunk for chunk in chunks]
                elif homotopy_type == "Fibration Type":
                    a0_base_vals = np.random.uniform(a0_range[0], a0_range[1], evaluations_per_test)
                    a1_fiber1_vals = np.random.uniform(a1_range[0], a1_range[1], evaluations_per_test)
                    a2_fiber2_vals = np.random.uniform(a2_range[0], a2_range[1], evaluations_per_test)
                    a3_fiber3_vals = np.random.uniform(a3_range[0], a3_range[1], evaluations_per_test)
                    chunk_size = 1000
                    chunks = list(zip(a0_base_vals, a1_fiber1_vals, a2_fiber2_vals, a3_fiber3_vals))
                    func = partial(simulate_homotopy, n, homotopy_type, perturbation)
                    input_data = [chunk for chunk in chunks]
                else:
                    raise ValueError(f"Unknown homotopy type: {homotopy_type}")

                # Map the simulation function across all chunks
                # Using tqdm for progress bar
                is_stable_results = list(tqdm(pool.imap(func, input_data), total=len(input_data), desc=f"Processing n={n}, type={homotopy_type}, perturbation={epsilon}"))

                # Aggregate results
                stable_count = sum(is_stable for is_stable, _ in is_stable_results)
                unstable_count = evaluations_per_test - stable_count
                stability_percentage = (stable_count / evaluations_per_test) * 100
                mean_value = np.mean([output for _, output in is_stable_results])
                std_deviation = np.std([output for _, output in is_stable_results])

                # Confidence Interval (95%)
                confidence_interval = (
                    mean_value - 1.96 * (std_deviation / math.sqrt(evaluations_per_test)),
                    mean_value + 1.96 * (std_deviation / math.sqrt(evaluations_per_test))
                )

                # Create a result instance
                result = AdvancedStabilityResult(
                    homotopy_level=n,
                    homotopy_type=homotopy_type,
                    perturbation_type=perturbation_type,
                    perturbation_value=epsilon,
                    total_evaluations=evaluations_per_test,
                    stable_count=stable_count,
                    unstable_count=unstable_count,
                    stability_percentage=stability_percentage,
                    mean_value=mean_value,
                    std_deviation=std_deviation,
                    confidence_interval=confidence_interval
                )

                results.append(result)
                print(f"Completed: Level={n}, Type={homotopy_type}, Perturbation={perturbation_type}={epsilon}")

    pool.close()
    pool.join()
    return results

# Function to generate perturbations, including adaptive scaling
def generate_advanced_perturbations(n_levels: List[int]) -> List[Tuple[str, float]]:
    perturbations = []
    for n in n_levels:
        # Fixed perturbations
        perturbations.extend([("Fixed_Positive", 0.5), ("Fixed_Negative", -0.5)])
        # Adaptive perturbations
        perturbations.append(("Adaptive_Positive", adaptive_scaling(n)))
        perturbations.append(("Adaptive_Negative", adaptive_scaling_negative(n)))
        # Non-linear perturbations
        perturbations.extend([("NonLinear_Positive", math.sin(0.5 * n)), ("NonLinear_Negative", math.cos(0.5 * n))])
    return perturbations

# Function to visualize results
def visualize_results(results: List[AdvancedStabilityResult]):
    # Convert results to DataFrame
    df = pd.DataFrame([result.__dict__ for result in results])

    # Plot stability percentage by homotopy level and type
    plt.figure(figsize=(12, 8))
    sns.barplot(x='homotopy_level', y='stability_percentage', hue='homotopy_type', data=df)
    plt.title('Stability Percentage by Homotopy Level and Type')
    plt.ylabel('Stability Percentage (%)')
    plt.xlabel('Homotopy Level')
    plt.legend(title='Homotopy Type')
    plt.tight_layout()
    plt.show()

    # Plot distribution of mean values
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='homotopy_type', y='mean_value', hue='homotopy_level', data=df)
    plt.title('Distribution of Mean Output Values by Homotopy Type and Level')
    plt.ylabel('Mean Output Value')
    plt.xlabel('Homotopy Type')
    plt.legend(title='Homotopy Level')
    plt.tight_layout()
    plt.show()

    # Plot confidence intervals
    plt.figure(figsize=(12, 8))
    for idx, row in df.iterrows():
        plt.errorbar(row['homotopy_level'], row['mean_value'],
                     yerr=(row['mean_value'] - row['confidence_interval'][0],
                           row['confidence_interval'][1] - row['mean_value']),
                     fmt='o', label=f"{row['homotopy_type']} n={row['homotopy_level']}" if idx < 3 else "")
    plt.title('Confidence Intervals of Mean Output Values')
    plt.ylabel('Mean Output Value')
    plt.xlabel('Homotopy Level')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

    # Save the DataFrame to a CSV for further analysis
    df.to_csv('advanced_homotopy_stability_results.csv', index=False)
    print("Results have been saved to 'advanced_homotopy_stability_results.csv'.")

# Main function to execute the advanced program
def main():
    # Define homotopy levels and types
    homotopy_levels = [1, 2, 3, 4, 5]
    homotopy_types = ["Loop Space", "Product Type", "Fibration Type"]

    # Generate advanced perturbations
    perturbations = generate_advanced_perturbations(homotopy_levels)

    # Define the number of evaluations per test
    evaluations_per_test = 100000  # Increased from 1000 to 100,000 for advanced testing

    # Perform advanced stability tests
    results = perform_advanced_stability_tests(
        homotopy_levels=homotopy_levels,
        homotopy_types=homotopy_types,
        perturbations=perturbations,
        evaluations_per_test=evaluations_per_test,
        a0_range=(0, 2 * math.pi),
        a1_range=(0, 2 * math.pi),
        a2_range=(0, 2 * math.pi),
        a3_range=(0, 2 * math.pi)
    )

    # Visualize the results
    visualize_results(results)

if __name__ == "__main__":
    main()
