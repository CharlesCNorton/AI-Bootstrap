import numpy as np
import gudhi as gd
import pandas as pd
from tqdm import tqdm

# Function to calculate Betti numbers with GUDHI
def calculate_betti_numbers_with_gudhi(num_vertices, dimension):
    """Use GUDHI to generate a simplicial complex and compute Betti numbers."""
    # Generate a random Rips complex
    rips_complex = gd.RipsComplex(points=np.random.rand(num_vertices, dimension), max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)

    # Compute persistence before calculating Betti numbers
    simplex_tree.compute_persistence()

    # Get Betti numbers using the correct method
    betti_numbers = simplex_tree.betti_numbers()
    return betti_numbers

# Function to simulate simplicial complexes and analyze with GUDHI
def simulate_complexes_with_gudhi(constant_factor, interaction_strength, enhanced=False):
    """Simulate simplicial complexes and collect statistics."""
    num_tests = 1000  # You can adjust the number of tests as needed
    results = []

    for _ in tqdm(range(num_tests)):
        # Random number of vertices and dimension for each test
        num_vertices = np.random.randint(10, 30)
        dimension = np.random.randint(2, 6)

        # Calculate Betti numbers with GUDHI
        betti_numbers = calculate_betti_numbers_with_gudhi(num_vertices, dimension)

        # Compute complexity and K_M using the betti numbers
        complexity = sum(betti_numbers)
        K_M = constant_factor * (sum(betti_numbers) + interaction_strength * num_vertices * dimension)

        # Determine if the computed K_M provides an upper bound
        bound_check = K_M >= complexity
        results.append({
            "num_vertices": num_vertices,
            "dimension": dimension,
            "betti_numbers": betti_numbers,
            "complexity": complexity,
            "K_M": K_M,
            "bound_check": bound_check
        })

    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    return df

# Function to perform parameter exploration and analysis
def parameter_exploration():
    constant_factors = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    interaction_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
    enhanced_options = [False, True]

    for constant_factor in constant_factors:
        for interaction_strength in interaction_strengths:
            for enhanced in enhanced_options:
                print(f"Running Batch with Parameters: Constant Factor={constant_factor}, Interaction Strength={interaction_strength}, Enhanced={enhanced}")
                df = simulate_complexes_with_gudhi(constant_factor, interaction_strength, enhanced=enhanced)

                # Summarize results
                total_tests = len(df)
                successful_checks = df['bound_check'].sum()
                failure_cases = total_tests - successful_checks
                success_rate = (successful_checks / total_tests) * 100

                print(f"=== Batch Simulation Results ===")
                print(f"Total Tests: {total_tests}")
                print(f"Successful Bound Checks: {successful_checks}")
                print(f"Failure Cases: {failure_cases}")
                print(f"Success Rate: {success_rate:.2f}%\n")

                # Analyzing failures
                failure_df = df[~df['bound_check']]
                if not failure_df.empty:
                    print(f"=== Failure Analysis Summary ===")
                    print(f"Number of Failures: {failure_cases}")
                    print(f"Average K_M in Failures: {failure_df['K_M'].mean():.2f}")
                    print(f"Average Complexity in Failures: {failure_df['complexity'].mean():.2f}")
                    print(f"Dimensions in Failures: {failure_df['dimension'].value_counts().to_dict()}\n")
                else:
                    print("No failures in this batch.\n")

# Run the parameter exploration
if __name__ == "__main__":
    parameter_exploration()
