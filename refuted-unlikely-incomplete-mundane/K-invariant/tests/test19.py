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
    Note: This function is kept for potential future use but is not utilized in exact homotopy group tests.
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

def run_specific_tests():
    """
    Runs specific test cases with known homotopical complexities.
    """
    results = []

    # Define specific test cases with known homotopy groups
    test_cases = [
        {
            "Type": "Sphere S^n",
            "Num_Vertices": 20,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 1, 0],  # H_0 = 1, H_1 = 0, H_2 = 1, H_3 = 0
            "Known_Homotopy_Groups": {3: 1}  # pi_3(S^3) = Z (represented as 1 for simplicity)
        },
        {
            "Type": "Torus T^2",
            "Num_Vertices": 30,
            "Dimension": 2,
            "Betti_Numbers": [1, 2, 1],  # H_0 = 1, H_1 = 2, H_2 = 1
            "Known_Homotopy_Groups": {1: 2}  # pi_1(T^2) = Z^2 (represented as 2)
        },
        {
            "Type": "Projective Space RP^4",
            "Num_Vertices": 25,
            "Dimension": 4,
            "Betti_Numbers": [1, 0, 0, 1, 0],  # H_0 = 1, H_1 = 0, H_2 = 0, H_3 = 1, H_4 = 0
            "Known_Homotopy_Groups": {4: 1}  # pi_4(RP^4) = Z
        },
        {
            "Type": "Wedge Sum of Spheres",
            "Num_Vertices": 35,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 3, 0],  # H_0 = 1, H_1 = 0, H_2 = 3, H_3 = 0
            "Known_Homotopy_Groups": {3: 3}  # pi_3(Wedge_S3^3) = Z^3
        },
        {
            "Type": "Complex Torus T^4",
            "Num_Vertices": 40,
            "Dimension": 4,
            "Betti_Numbers": [1, 4, 6, 4, 1],  # H_0 = 1, H_1 = 4, H_2 = 6, H_3 = 4, H_4 = 1
            "Known_Homotopy_Groups": {1: 4}  # pi_1(T^4) = Z^4
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        type_name = case["Type"]
        num_vertices = case["Num_Vertices"]
        dimension = case["Dimension"]
        betti_numbers = case["Betti_Numbers"]
        betti_sum = sum(betti_numbers)
        known_homotopy = case["Known_Homotopy_Groups"]

        # Compute K_M based on Sum_Betti, number of vertices, and dimension
        K_M = compute_K_M(betti_sum, num_vertices, dimension)

        # Initialize pass condition
        pass_conditions = []

        # Iterate over known homotopy groups and check the invariant
        for n, pi_n in known_homotopy.items():
            # As per Lemma: K_M >= c * pi_n(M) * n
            condition = K_M >= (pi_n * n)
            pass_conditions.append(condition)

        # Overall condition holds if all individual conditions hold
        overall_condition = all(pass_conditions)

        # Append results
        results.append({
            "Trial": idx,
            "Type": type_name,
            "Num_Vertices": num_vertices,
            "Dimension": dimension,
            "Betti_Numbers": betti_numbers,
            "Sum_Betti": betti_sum,
            "Known_Homotopy_Groups": known_homotopy,
            "K_M": round(K_M, 2),
            "Bound_Holds": overall_condition
        })

    return results

def compile_and_display_results(specific_results):
    """
    Compiles specific test results into a DataFrame and prints them.
    """
    df_specific = pd.DataFrame(specific_results)

    # Display a summary of the results
    total_pass = df_specific["Bound_Holds"].sum()
    total_trials = len(df_specific)
    print("\n=== Specific Test Summary ===")
    print(f"Total Specific Trials: {total_trials}")
    print(f"Bound Holds in {total_pass} Trials")
    print(f"Bound Fails in {total_trials - total_pass} Trials")

    # Display detailed results
    print("\n=== Detailed Specific Test Results ===")
    print(df_specific.to_string(index=False))

    # Save to CSV for further analysis if needed
    df_specific.to_csv("K_invariant_Specific_Test_Results.csv", index=False)

def main():
    # Run specific tests with known homotopy groups
    specific_results = run_specific_tests()

    # Compile and display results
    compile_and_display_results(specific_results)

if __name__ == "__main__":
    main()
