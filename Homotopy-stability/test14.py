import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Tuple

# Define a dataclass to store the results of each test
@dataclass
class StabilityResult:
    homotopy_level: int
    homotopy_type: str
    perturbation: float
    total_evaluations: int
    stable: int
    unstable: int
    stability_percentage: float
    mean_value: float
    std_deviation: float

# Define the perturbation scaling function for adaptive scaling
def adaptive_scaling(n: int) -> float:
    return 1.0 / (1 + n)

# Define the perturbation scaling function for adaptive scaling (negative)
def adaptive_scaling_negative(n: int) -> float:
    return -1.0 / (1 + n)

# Define the perturbation function P(a0, epsilon)
def P(a0: float, epsilon: float) -> float:
    return a0 + epsilon

# Define the Loop Space L_n(a0, epsilon)
def loop_space(n: int, a0: float, epsilon: float) -> float:
    try:
        term1 = ((a0 + P(a0, epsilon)) / 2) ** (1 / n)
    except ZeroDivisionError:
        term1 = float('inf')
    term2 = math.cos(n * (a0 + epsilon))
    return term1 + term2

# Define the Product Type P_n(a0_1, a0_2, epsilon)
def product_type(n: int, a0_1: float, a0_2: float, epsilon: float) -> float:
    try:
        term1 = (a0_1 + epsilon) ** (1 / n)
    except ZeroDivisionError:
        term1 = float('inf')
    term2 = math.cos(n * (a0_1 + epsilon))
    term3 = (a0_2 - epsilon) ** (1 / n)
    term4 = math.sin(n * (a0_2 - epsilon))
    return (term1 + term2 + term3 + term4) / 2

# Define the Fibration Type F_n(a0_base, a0_fiber1, a0_fiber2, epsilon)
def fibration_type(n: int, a0_base: float, a0_fiber1: float, a0_fiber2: float, epsilon: float) -> float:
    # Define cup products; for simplicity, set them to 0.0
    cup_product1 = 0.0
    cup_product2 = 0.0

    try:
        term_base = (a0_base + epsilon) ** (1 / n) + math.cos(n * a0_base)
    except ZeroDivisionError:
        term_base = float('inf')

    try:
        term_fiber1 = (a0_fiber1 + 0.5 * epsilon) ** (1 / (n + 1)) + math.sin(n * a0_fiber1) + cup_product1
    except ZeroDivisionError:
        term_fiber1 = float('inf')

    try:
        term_fiber2 = (a0_fiber2 + 0.25 * epsilon) ** (1 / (n + 2)) + math.sin(n * a0_fiber2) + cup_product2
    except ZeroDivisionError:
        term_fiber2 = float('inf')

    avg_fiber = (term_fiber1 + term_fiber2) / 2
    return (term_base + avg_fiber) / 2

# Function to perform stability tests
def perform_stability_tests(
    n_levels: List[int],
    homotopy_types: List[str],
    perturbations: List[float],
    evaluations_per_test: int = 1000,
    a0_range: Tuple[float, float] = (0, 2 * math.pi),
    delta: float = 2.0
) -> List[StabilityResult]:
    results = []

    for n in n_levels:
        for homotopy_type in homotopy_types:
            for epsilon in perturbations:
                stable_count = 0
                unstable_count = 0
                outputs = []

                for _ in range(evaluations_per_test):
                    # Sample a0 uniformly from the specified range
                    a0 = np.random.uniform(a0_range[0], a0_range[1])

                    # Depending on homotopy type, compute the output
                    if homotopy_type == "Loop Space":
                        output = loop_space(n, a0, epsilon)
                    elif homotopy_type == "Product Type":
                        # For Product Type, sample a0_1 and a0_2
                        a0_1 = np.random.uniform(a0_range[0], a0_range[1])
                        a0_2 = np.random.uniform(a0_range[0], a0_range[1])
                        output = product_type(n, a0_1, a0_2, epsilon)
                    elif homotopy_type == "Fibration Type":
                        # For Fibration Type, sample a0_base, a0_fiber1, a0_fiber2
                        a0_base = np.random.uniform(a0_range[0], a0_range[1])
                        a0_fiber1 = np.random.uniform(a0_range[0], a0_range[1])
                        a0_fiber2 = np.random.uniform(a0_range[0], a0_range[1])
                        output = fibration_type(n, a0_base, a0_fiber1, a0_fiber2, epsilon)
                    else:
                        raise ValueError(f"Unknown homotopy type: {homotopy_type}")

                    outputs.append(output)

                    # Check stability condition
                    if abs(output) < delta:
                        stable_count += 1
                    else:
                        unstable_count += 1

                # Calculate statistics
                mean_val = np.mean(outputs)
                std_dev = np.std(outputs)
                stability_pct = (stable_count / evaluations_per_test) * 100

                # Create a StabilityResult instance and add to results
                result = StabilityResult(
                    homotopy_level=n,
                    homotopy_type=homotopy_type,
                    perturbation=epsilon,
                    total_evaluations=evaluations_per_test,
                    stable=stable_count,
                    unstable=unstable_count,
                    stability_percentage=stability_pct,
                    mean_value=mean_val,
                    std_deviation=std_dev
                )

                results.append(result)
                print(f"Completed: Level={n}, Type={homotopy_type}, Epsilon={epsilon}")

    return results

# Function to generate perturbations, including adaptive scaling
def generate_perturbations(n_levels: List[int]) -> List[float]:
    perturbations = []
    for n in n_levels:
        # Fixed perturbations
        perturbations.extend([0.5, -0.5])
        # Adaptive perturbations
        perturbations.append(adaptive_scaling(n))
        perturbations.append(adaptive_scaling_negative(n))
    return perturbations

# Main function to execute the tests and display results
def main():
    # Define homotopy levels and types
    homotopy_levels = [1, 2, 3, 4, 5]
    homotopy_types = ["Loop Space", "Product Type", "Fibration Type"]

    # Generate perturbations
    perturbations = generate_perturbations(homotopy_levels)

    # Perform stability tests
    results = perform_stability_tests(
        n_levels=homotopy_levels,
        homotopy_types=homotopy_types,
        perturbations=perturbations,
        evaluations_per_test=1000,
        a0_range=(0, 2 * math.pi),
        delta=2.0
    )

    # Convert results to a DataFrame for better visualization
    df = pd.DataFrame([result.__dict__ for result in results])

    # Display the results
    pd.set_option('display.max_rows', None)
    print("\nStability Test Results:")
    print(df)

if __name__ == "__main__":
    main()
