import numpy as np
import gudhi as gd
import random
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans

# Define the function to calculate curvature index K_M with intrinsic parameters
def calculate_K_M(c_values, x_values, c_0, interaction_matrix=None, exponential=False):
    linear_combination = np.sum(c_values * x_values) + c_0
    quadratic_term = linear_combination ** 2
    sine_term = np.sin(linear_combination)

    # Cross-dimensional interaction terms
    if interaction_matrix is not None:
        cross_dimensional_interaction = np.sum(interaction_matrix)  # Sum of interaction matrix entries
    else:
        cross_dimensional_interaction = 0

    if exponential:
        exponential_term = np.exp(0.1 * linear_combination)
        K_M = quadratic_term + sine_term + exponential_term + cross_dimensional_interaction
    else:
        K_M = quadratic_term + sine_term + cross_dimensional_interaction

    return K_M

# Monte Carlo Simulation Parameters
batch_size = 10000  # Increased per batch for more detailed exploration
failure_data = []

# Batch runs with intrinsic parameter exploration
parameter_batches = [
    {"constant_factor": 1.5, "interaction_strength": 0.1},
    {"constant_factor": 2.0, "interaction_strength": 0.3},
    {"constant_factor": 1.8, "interaction_strength": 0.2},
    {"constant_factor": 1.6, "interaction_strength": 0.5},
    {"constant_factor": 1.7, "interaction_strength": 0.1},
]

# Loop through parameter batches to explore their effects
for param_set in parameter_batches:
    constant_factor = param_set["constant_factor"]
    interaction_strength = param_set["interaction_strength"]

    print(f"\nRunning Batch with Parameters: Constant Factor={constant_factor}, Interaction Strength={interaction_strength}")

    successful_bounds = 0  # Reset for each batch
    bound_check_results = []

    for test_num in tqdm(range(batch_size)):
        # Step 1: Generate a random simplicial complex with higher dimensionality
        complex = gd.SimplexTree()
        num_vertices = random.randint(10, 25)  # Number of vertices for complexity
        max_dimension = random.randint(2, 6)   # Higher dimensions to test robustness

        # Insert random simplices
        for _ in range(60):  # Insert more simplices to increase density
            simplex = random.sample(range(num_vertices), random.randint(1, max_dimension + 1))
            complex.insert(simplex)

        # Step 2: Compute persistent homology
        complex.compute_persistence()
        betti_numbers = complex.betti_numbers()
        complexity = sum(betti_numbers)

        # Step 3: Calculate K_M with systematic parameter variation
        c_values = np.random.uniform(-5, 5, num_vertices)  # Allow negative and positive values
        x_values = np.random.uniform(-10, 15, num_vertices)  # Wider range for x_values
        c_0 = np.random.uniform(-5, 5)

        # Cross-Dimensional Interaction Matrix (based on Betti numbers)
        interaction_matrix = np.outer(betti_numbers, betti_numbers) * interaction_strength

        # Calculate K_M
        K_M = calculate_K_M(c_values, x_values, c_0, interaction_matrix=interaction_matrix, exponential=True)

        # Step 4: Check the hypothesis
        bound_check = K_M >= constant_factor * complexity
        bound_check_results.append(bound_check)

        if bound_check:
            successful_bounds += 1
        else:
            # Record detailed information about the failure
            failure_data.append({
                'Test Number': test_num,
                'K_M': K_M,
                'Complexity': complexity,
                'Max_Dimension': max_dimension,
                'Vertices': num_vertices,
                'Constant Factor': constant_factor,
                'Interaction Strength': interaction_strength,
                'Betti Numbers': betti_numbers
            })

    # Summary of Batch Results
    failure_cases = batch_size - successful_bounds
    success_rate = successful_bounds / batch_size * 100

    print("\n=== Batch Simulation Results ===")
    print(f"Total Tests: {batch_size}")
    print(f"Successful Bound Checks: {successful_bounds}")
    print(f"Failure Cases: {failure_cases}")
    print(f"Success Rate: {success_rate:.2f}%")

# === Failure Analysis After Parameter Exploration ===
if len(failure_data) > 0:
    print("\n=== Failure Analysis Summary Across Batches ===")
    df = pd.DataFrame(failure_data)

    # Display summary statistics
    print("\nNumber of Failures by Parameter Set:")
    parameter_groups = df.groupby(['Constant Factor', 'Interaction Strength']).size()
    for key, value in parameter_groups.items():
        print(f"Constant Factor={key[0]}, Interaction Strength={key[1]}: {value} failures")

    # Clustering Analysis to Group Failures
    print("\n=== Clustering Analysis of Failures ===")
    if len(df) > 10:
        # Perform K-means clustering on failures based on parameters and K_M
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[['K_M', 'Complexity', 'Max_Dimension', 'Vertices']])

        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            print(f"\nCluster {cluster + 1} ({len(cluster_data)} elements):")
            print(f"  Average K_M: {cluster_data['K_M'].mean():.2f}")
            print(f"  Average Complexity: {cluster_data['Complexity'].mean():.2f}")
            print(f"  Average Dimension: {cluster_data['Max_Dimension'].mean():.2f}")
            print(f"  Average Vertices: {cluster_data['Vertices'].mean():.2f}")
            print(f"  Common Parameter Set: Constant Factor={cluster_data['Constant Factor'].mode()[0]}, "
                  f"Interaction Strength={cluster_data['Interaction Strength'].mode()[0]}")
    else:
        print("Not enough failures for clustering analysis.")
else:
    print("\nSUCCESS: No failures across all parameter batches.")
