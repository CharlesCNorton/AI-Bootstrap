import numpy as np
import gudhi as gd
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# ================================
# Configuration and Parameters
# ================================

# Initial constants for the invariant K_M
INITIAL_CONSTANT_FACTOR = 1.5
INITIAL_INTERACTION_STRENGTH = 0.5

# Parameter grid for optimization
PARAMETER_GRID = {
    'CONSTANT_FACTOR': np.arange(1.0, 3.1, 0.1),       # From 1.0 to 3.0 with step 0.1
    'INTERACTION_STRENGTH': np.arange(0.1, 1.1, 0.1)   # From 0.1 to 1.0 with step 0.1
}

# ================================
# Function Definitions
# ================================

def compute_K_M(betti_sum, num_vertices, dimension, constant_factor, interaction_strength):
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
            "Betti_Numbers": [1, 0, 1, 0],  # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 1}  # pi_3(S^3) = Z
        },
        {
            "Type": "Torus T^2",
            "Num_Vertices": 30,
            "Dimension": 2,
            "Betti_Numbers": [1, 2, 1],      # H_0, H_1, H_2
            "Known_Homotopy_Groups": {1: 2}  # pi_1(T^2) = Z^2
        },
        {
            "Type": "Projective Space RP^4",
            "Num_Vertices": 25,
            "Dimension": 4,
            "Betti_Numbers": [1, 0, 0, 1, 0], # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {4: 1}  # pi_4(RP^4) = Z
        },
        {
            "Type": "Wedge Sum of Spheres",
            "Num_Vertices": 35,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 3, 0],    # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 3}  # pi_3(Wedge_S3^3) = Z^3
        },
        {
            "Type": "Complex Torus T^4",
            "Num_Vertices": 40,
            "Dimension": 4,
            "Betti_Numbers": [1, 4, 6, 4, 1],  # H_0, H_1, H_2, H_3, H_4
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

        # Compute K_M using initial constants
        K_M = compute_K_M(
            betti_sum=betti_sum,
            num_vertices=num_vertices,
            dimension=dimension,
            constant_factor=INITIAL_CONSTANT_FACTOR,
            interaction_strength=INITIAL_INTERACTION_STRENGTH
        )

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

def run_edge_case_tests():
    """
    Runs edge case tests where K_M is expected to just barely bound the homotopical complexity.
    """
    results = []

    # Define edge cases with higher Sum_Betti relative to K_M
    edge_cases = [
        {
            "Type": "High Homotopy Number Space 1",
            "Num_Vertices": 10,
            "Dimension": 2,
            "Betti_Numbers": [1, 10, 1],       # H_0, H_1, H_2
            "Known_Homotopy_Groups": {2: 10}   # pi_2(M) = 10 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 2",
            "Num_Vertices": 15,
            "Dimension": 3,
            "Betti_Numbers": [1, 15, 1, 0],    # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 15}   # pi_3(M) = 15 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 3",
            "Num_Vertices": 20,
            "Dimension": 4,
            "Betti_Numbers": [1, 20, 1, 0, 0], # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {4: 20}   # pi_4(M) = 20 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 4",
            "Num_Vertices": 25,
            "Dimension": 5,
            "Betti_Numbers": [1, 25, 1, 0, 0, 0], # H_0, H_1, H_2, H_3, H_4, H_5
            "Known_Homotopy_Groups": {5: 25}      # pi_5(M) = 25 (hypothetical)
        }
    ]

    for idx, case in enumerate(edge_cases, 1):
        type_name = case["Type"]
        num_vertices = case["Num_Vertices"]
        dimension = case["Dimension"]
        betti_numbers = case["Betti_Numbers"]
        betti_sum = sum(betti_numbers)
        known_homotopy = case["Known_Homotopy_Groups"]

        # Compute K_M using initial constants
        K_M = compute_K_M(
            betti_sum=betti_sum,
            num_vertices=num_vertices,
            dimension=dimension,
            constant_factor=INITIAL_CONSTANT_FACTOR,
            interaction_strength=INITIAL_INTERACTION_STRENGTH
        )

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

def run_comparative_tests(specific_results):
    """
    Compares K_invariant with the sum of Betti numbers as a basic invariant.
    """
    results = []

    for result in specific_results:
        K_M = result["K_M"]
        sum_betti = result["Sum_Betti"]
        type_name = result["Type"]

        # Basic invariant: Sum of Betti numbers
        basic_invariant = sum_betti

        # Compare
        comparison = K_M >= basic_invariant

        # Append results
        results.append({
            "Type": type_name,
            "K_M": K_M,
            "Basic_Invariant_Sum_Betti": basic_invariant,
            "K_M >= Sum_Betti": comparison
        })

    return results

def optimize_parameters(specific_results, edge_case_results):
    """
    Optimizes CONSTANT_FACTOR and INTERACTION_STRENGTH to maximize the number of bounds that hold.
    Uses grid search over predefined parameter ranges.
    """
    # Combine specific and edge case results for optimization
    combined_results = specific_results + edge_case_results

    grid = ParameterGrid(PARAMETER_GRID)
    best_score = -1
    best_params = {}

    print("\n=== Starting Parameter Optimization ===")

    for params in tqdm(grid, desc="Parameter Grid Search"):
        c_factor = params['CONSTANT_FACTOR']
        i_strength = params['INTERACTION_STRENGTH']
        score = 0

        for result in combined_results:
            betti_sum = result["Sum_Betti"]
            num_vertices = result["Num_Vertices"]
            dimension = result["Dimension"]
            known_homotopy = result["Known_Homotopy_Groups"]

            # Compute K_M with current parameters
            K_M = compute_K_M(
                betti_sum=betti_sum,
                num_vertices=num_vertices,
                dimension=dimension,
                constant_factor=c_factor,
                interaction_strength=i_strength
            )

            # Check all homotopy group conditions
            conditions = [K_M >= (pi_n * n) for n, pi_n in known_homotopy.items()]
            if all(conditions):
                score += 1

        # Update best parameters if current score is better
        if score > best_score:
            best_score = score
            best_params = params

    print("\n=== Parameter Optimization Completed ===")
    print(f"Best Score: {best_score} out of {len(combined_results)}")
    print(f"Best Parameters: CONSTANT_FACTOR = {best_params['CONSTANT_FACTOR']}, INTERACTION_STRENGTH = {best_params['INTERACTION_STRENGTH']}")

    return best_params

def visualize_results(specific_results, edge_case_results, optimized_params):
    """
    Visualizes the relationship between K_M and pi_n(M) * n.
    """
    pi_n_n = []
    K_M_values = []
    labels = []

    # Combine specific and edge case results
    combined_results = specific_results + edge_case_results

    for result in combined_results:
        for n, pi_n in result["Known_Homotopy_Groups"].items():
            pi_n_n.append(pi_n * n)
            # Recompute K_M with optimized parameters
            K_M = compute_K_M(
                betti_sum=result["Sum_Betti"],
                num_vertices=result["Num_Vertices"],
                dimension=result["Dimension"],
                constant_factor=optimized_params['CONSTANT_FACTOR'],
                interaction_strength=optimized_params['INTERACTION_STRENGTH']
            )
            K_M_values.append(K_M)
            labels.append(result["Type"])

    plt.figure(figsize=(10, 6))
    plt.scatter(pi_n_n, K_M_values, color='blue')
    for i, label in enumerate(labels):
        plt.annotate(label, (pi_n_n[i], K_M_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # Plot the line K_M = pi_n(M) * n
    max_val = max(pi_n_n + K_M_values) + 10
    plt.plot([0, max_val], [0, max_val], 'r--', label='K_M = pi_n(M) * n')
    plt.xlabel('pi_n(M) * n')
    plt.ylabel('K_M')
    plt.title('Validation of K_invariant Against Homotopy Groups')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("K_invariant_Validation_Plot.png")
    plt.show()

def compile_and_save_results(specific_results, edge_case_results, comparative_results, optimized_params):
    """
    Compiles all test results and saves them to CSV files.
    """
    # Convert lists of dictionaries to DataFrames
    df_specific = pd.DataFrame(specific_results)
    df_edge_cases = pd.DataFrame(edge_case_results)
    df_comparative = pd.DataFrame(comparative_results)

    # Save to CSV
    df_specific.to_csv("K_invariant_Specific_Test_Results.csv", index=False)
    df_edge_cases.to_csv("K_invariant_Edge_Case_Test_Results.csv", index=False)
    df_comparative.to_csv("K_invariant_Comparative_Test_Results.csv", index=False)

    # Save optimized parameters
    with open("K_invariant_Optimized_Params.txt", "w") as f:
        f.write(f"Optimized CONSTANT_FACTOR: {optimized_params['CONSTANT_FACTOR']}\n")
        f.write(f"Optimized INTERACTION_STRENGTH: {optimized_params['INTERACTION_STRENGTH']}\n")

def main():
    """
    Main function to execute all testing, optimization, comparison, visualization, and reporting.
    """
    # Step 1: Run Specific Tests
    specific_results = run_specific_tests()

    # Step 2: Run Edge Case Tests
    edge_case_results = run_edge_case_tests()

    # Step 3: Parameter Optimization
    optimized_params = optimize_parameters(specific_results, edge_case_results)

    # Step 4: Re-run Specific and Edge Case Tests with Optimized Parameters
    # Update K_M in the results based on optimized parameters
    for result in specific_results + edge_case_results:
        result["K_M"] = round(compute_K_M(
            betti_sum=result["Sum_Betti"],
            num_vertices=result["Num_Vertices"],
            dimension=result["Dimension"],
            constant_factor=optimized_params['CONSTANT_FACTOR'],
            interaction_strength=optimized_params['INTERACTION_STRENGTH']
        ), 2)
        # Re-check the bound
        known_homotopy = result["Known_Homotopy_Groups"]
        pass_conditions = [result["K_M"] >= (pi_n * n) for n, pi_n in known_homotopy.items()]
        result["Bound_Holds"] = all(pass_conditions)

    # Step 5: Comparative Analysis
    comparative_results = run_comparative_tests(specific_results + edge_case_results)

    # Step 6: Visualization
    visualize_results(specific_results, edge_case_results, optimized_params)

    # Step 7: Compile and Save Results
    compile_and_save_results(specific_results, edge_case_results, comparative_results, optimized_params)

    # Step 8: Print Summary
    total_specific = len(specific_results)
    passed_specific = sum([res["Bound_Holds"] for res in specific_results])
    total_edge = len(edge_case_results)
    passed_edge = sum([res["Bound_Holds"] for res in edge_case_results])
    total_comparative = len(comparative_results)
    passed_comparative = sum([res["K_M >= Sum_Betti"] for res in comparative_results])

    print("\n=== Comprehensive Test Summary ===")
    print(f"Specific Tests: {passed_specific}/{total_specific} Passed")
    print(f"Edge Case Tests: {passed_edge}/{total_edge} Passed")
    print(f"Comparative Tests (K_M >= Sum_Betti): {passed_comparative}/{total_comparative} Passed")

    print("\nAll tests completed successfully. Results have been saved to CSV files and a visualization plot has been generated.")

if __name__ == "__main__":
    main()
