import numpy as np
import gudhi as gd
import pandas as pd
from tqdm import tqdm

# Define constants for the invariant K_M
CONSTANT_FACTOR = 1.5         # Adjustable based on empirical observations
INTERACTION_STRENGTH = 0.5    # Adjustable based on empirical observations

def calculate_betti_numbers(num_vertices, dimension):
    """
    Generates a random Rips complex and computes its Betti numbers up to the specified dimension.
    """
    # Generate random points in Euclidean space
    points = np.random.rand(num_vertices, dimension)

    # Create a Rips complex with a maximum edge length (can be adjusted)
    rips_complex = gd.RipsComplex(points=points, max_edge_length=2.0)

    # Create a simplex tree up to the specified dimension
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)

    # Compute persistence (optional, can be omitted if only Betti numbers are needed)
    simplex_tree.compute_persistence()

    # Retrieve Betti numbers
    betti_numbers = simplex_tree.betti_numbers()

    # Ensure that the Betti numbers list has entries up to the specified dimension
    # If not, pad with zeros
    while len(betti_numbers) < (dimension + 1):
        betti_numbers.append(0)

    return betti_numbers

def compute_K_M(betti_sum, num_vertices, dimension, constant_factor=CONSTANT_FACTOR, interaction_strength=INTERACTION_STRENGTH):
    """
    Computes the curvature index K_M based on the sum of Betti numbers, number of vertices, and dimension.
    """
    K_M = constant_factor * (betti_sum + interaction_strength * num_vertices * dimension)
    return K_M

def run_random_tests(num_trials=100):
    """
    Runs multiple random trials to test whether K_M bounds the homotopical complexity as claimed.
    """
    results = []

    for trial in tqdm(range(1, num_trials + 1), desc="Running Random Trials"):
        # Randomly choose number of vertices and dimension for each trial
        num_vertices = np.random.randint(10, 50)    # Number of vertices between 10 and 50
        dimension = np.random.randint(2, 6)         # Dimension between 2 and 5

        # Calculate Betti numbers
        betti_numbers = calculate_betti_numbers(num_vertices, dimension)
        betti_sum = sum(betti_numbers)

        # Compute K_M
        K_M = compute_K_M(betti_sum, num_vertices, dimension)

        # Define the invariant's upper bound condition
        # As per Lemma 1: K_M >= c * pi_n(M) * n
        # Here, we use betti_sum as a proxy for pi_n(M)
        # and 'n' as the dimension
        bound_condition = K_M >= (betti_sum * dimension)

        # Append results
        results.append({
            "Trial": trial,
            "Type": "Random",
            "Num_Vertices": num_vertices,
            "Dimension": dimension,
            "Betti_Numbers": betti_numbers,
            "Sum_Betti": betti_sum,
            "K_M": round(K_M, 2),
            "Bound_Holds": bound_condition
        })

    return results

def run_specific_tests():
    """
    Runs specific test cases with known homotopical complexities.
    """
    results = []

    # Define specific test cases
    test_cases = [
        {"Type": "Sphere", "Num_Vertices": 20, "Dimension": 3, "Betti_Numbers": [1, 0, 1, 0]},
        {"Type": "Torus", "Num_Vertices": 30, "Dimension": 2, "Betti_Numbers": [1, 2, 1]},
        {"Type": "Projective Space", "Num_Vertices": 25, "Dimension": 4, "Betti_Numbers": [1, 0, 0, 1, 0]},
        {"Type": "Wedge Sum of Spheres", "Num_Vertices": 35, "Dimension": 3, "Betti_Numbers": [1, 0, 3, 0]},
        {"Type": "Complex Torus", "Num_Vertices": 40, "Dimension": 4, "Betti_Numbers": [1, 4, 6, 4, 1]}
    ]

    for idx, case in enumerate(test_cases, 1):
        type_name = case["Type"]
        num_vertices = case["Num_Vertices"]
        dimension = case["Dimension"]
        betti_numbers = case["Betti_Numbers"]
        betti_sum = sum(betti_numbers)

        # Compute K_M
        K_M = compute_K_M(betti_sum, num_vertices, dimension)

        # Define the invariant's upper bound condition
        bound_condition = K_M >= (betti_sum * dimension)

        # Append results
        results.append({
            "Trial": idx,
            "Type": type_name,
            "Num_Vertices": num_vertices,
            "Dimension": dimension,
            "Betti_Numbers": betti_numbers,
            "Sum_Betti": betti_sum,
            "K_M": round(K_M, 2),
            "Bound_Holds": bound_condition
        })

    return results

def compile_results(random_results, specific_results):
    """
    Compiles random and specific test results into a single DataFrame and prints them.
    """
    df_random = pd.DataFrame(random_results)
    df_specific = pd.DataFrame(specific_results)

    # Combine the DataFrames
    df_all = pd.concat([df_random, df_specific], ignore_index=True)

    # Display a summary of the results
    total_pass = df_all["Bound_Holds"].sum()
    total_trials = len(df_all)
    print("\n=== Test Summary ===")
    print(f"Total Trials: {total_trials}")
    print(f"Bound Holds in {total_pass} Trials")
    print(f"Bound Fails in {total_trials - total_pass} Trials")

    # Display detailed results
    print("\n=== Detailed Results ===")
    print(df_all.to_string(index=False))

    # Save to CSV for further analysis if needed
    df_all.to_csv("K_invariant_Test_Results.csv", index=False)

def main():
    # Run random tests
    random_results = run_random_tests(num_trials=100)

    # Run specific tests
    specific_results = run_specific_tests()

    # Compile and display results
    compile_results(random_results, specific_results)

if __name__ == "__main__":
    main()
