import numpy as np
import gudhi as gd
import random
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans

# Function to calculate the basic curvature index K_M
def calculate_basic_K_M(c_values, x_values, c_0, exponential=False):
    linear_combination = np.sum(c_values * x_values) + c_0
    quadratic_term = linear_combination ** 2
    sine_term = np.sin(linear_combination)

    if exponential:
        exponential_term = np.exp(0.1 * linear_combination)
        K_M = quadratic_term + sine_term + exponential_term
    else:
        K_M = quadratic_term + sine_term

    return K_M

# Function to calculate the enhanced curvature index K_M
def calculate_enhanced_K_M(c_values, x_values, c_0, betti_numbers, interaction_matrix=None, exponential=False):
    linear_combination = np.sum(c_values * x_values) + c_0
    quadratic_term = linear_combination ** 2
    sine_term = np.sin(linear_combination)

    # Cross-dimensional interaction terms
    if interaction_matrix is not None:
        cross_dimensional_interaction = np.sum(interaction_matrix)
    else:
        cross_dimensional_interaction = 0

    # Enhanced interaction term involving higher-order products of Betti numbers
    betti_product_term = np.sum([b ** 2 for b in betti_numbers])
    betti_cubic_term = np.sum([b ** 3 for b in betti_numbers])

    # Fourier components to capture oscillatory behavior in complex topologies
    fourier_component = np.sum([np.sin((i + 1) * linear_combination) for i in range(len(betti_numbers))])

    if exponential:
        exponential_term = np.exp(0.1 * linear_combination)
        K_M = (
            quadratic_term
            + sine_term
            + exponential_term
            + cross_dimensional_interaction
            + betti_product_term
            + betti_cubic_term
            + fourier_component
        )
    else:
        K_M = (
            quadratic_term
            + sine_term
            + cross_dimensional_interaction
            + betti_product_term
            + betti_cubic_term
            + fourier_component
        )

    return K_M

# Monte Carlo Simulation Parameters
batch_size = 10000
failure_data = []

# Parameter sets for extended exploration
parameter_batches = [
    {"constant_factor": f, "interaction_strength": s}
    for f in np.arange(1.5, 3.1, 0.1)  # Constant factors from 1.5 to 3.0
    for s in np.arange(0.1, 1.1, 0.1)  # Interaction strengths from 0.1 to 1.0
]

# Loop through each parameter set and run tests for different configurations
for param_set in parameter_batches:
    constant_factor = param_set["constant_factor"]
    interaction_strength = param_set["interaction_strength"]

    # Configuration 1: Basic K_M with extended parameters
    print(f"\nRunning Basic K_M with Parameters: Constant Factor={constant_factor}, Interaction Strength={interaction_strength}")
    successful_bounds = 0
    for test_num in tqdm(range(batch_size)):
        # Randomly generate simplicial complex
        complex = gd.SimplexTree()
        num_vertices = random.randint(10, 25)
        max_dimension = random.randint(2, 6)

        for _ in range(60):
            simplex = random.sample(range(num_vertices), random.randint(1, max_dimension + 1))
            complex.insert(simplex)

        complex.compute_persistence()
        betti_numbers = complex.betti_numbers()
        complexity = sum(betti_numbers)

        # Calculate basic K_M
        c_values = np.random.uniform(-5, 5, num_vertices)
        x_values = np.random.uniform(-10, 15, num_vertices)
        c_0 = np.random.uniform(-5, 5)

        K_M = calculate_basic_K_M(c_values, x_values, c_0, exponential=True)

        # Check hypothesis
        bound_check = K_M >= constant_factor * complexity

        if bound_check:
            successful_bounds += 1
        else:
            failure_data.append({
                'Configuration': 'Basic K_M',
                'Test Number': test_num,
                'K_M': K_M,
                'Complexity': complexity,
                'Max_Dimension': max_dimension,
                'Vertices': num_vertices,
                'Constant Factor': constant_factor,
                'Interaction Strength': interaction_strength,
                'Betti Numbers': betti_numbers
            })

    # Summary of Basic K_M Batch Results
    print("\n=== Basic K_M Batch Simulation Results ===")
    print(f"Total Tests: {batch_size}")
    print(f"Successful Bound Checks: {successful_bounds}")
    print(f"Failure Cases: {batch_size - successful_bounds}")
    print(f"Success Rate: {successful_bounds / batch_size * 100:.2f}%")

    # Configuration 2: Enhanced K_M with same parameters
    print(f"\nRunning Enhanced K_M with Parameters: Constant Factor={constant_factor}, Interaction Strength={interaction_strength}")
    successful_bounds = 0
    for test_num in tqdm(range(batch_size)):
        # Randomly generate simplicial complex
        complex = gd.SimplexTree()
        num_vertices = random.randint(10, 25)
        max_dimension = random.randint(2, 6)

        for _ in range(60):
            simplex = random.sample(range(num_vertices), random.randint(1, max_dimension + 1))
            complex.insert(simplex)

        complex.compute_persistence()
        betti_numbers = complex.betti_numbers()
        complexity = sum(betti_numbers)

        # Cross-Dimensional Interaction Matrix
        interaction_matrix = np.outer(betti_numbers, betti_numbers) * interaction_strength

        # Calculate enhanced K_M
        K_M = calculate_enhanced_K_M(
            c_values,
            x_values,
            c_0,
            betti_numbers,
            interaction_matrix=interaction_matrix,
            exponential=True
        )

        # Check hypothesis
        bound_check = K_M >= constant_factor * complexity

        if bound_check:
            successful_bounds += 1
        else:
            failure_data.append({
                'Configuration': 'Enhanced K_M',
                'Test Number': test_num,
                'K_M': K_M,
                'Complexity': complexity,
                'Max_Dimension': max_dimension,
                'Vertices': num_vertices,
                'Constant Factor': constant_factor,
                'Interaction Strength': interaction_strength,
                'Betti Numbers': betti_numbers
            })

    # Summary of Enhanced K_M Batch Results
    print("\n=== Enhanced K_M Batch Simulation Results ===")
    print(f"Total Tests: {batch_size}")
    print(f"Successful Bound Checks: {successful_bounds}")
    print(f"Failure Cases: {batch_size - successful_bounds}")
    print(f"Success Rate: {successful_bounds / batch_size * 100:.2f}%")

# === Clustering Analysis of Failures ===
if len(failure_data) > 0:
    print("\n=== Failure Analysis Summary Across All Batches ===")
    df = pd.DataFrame(failure_data)

    # Display summary statistics
    print("\nNumber of Failures by Configuration and Parameter Set:")
    configuration_groups = df.groupby(['Configuration', 'Constant Factor', 'Interaction Strength']).size()
    for key, value in configuration_groups.items():
        print(f"Configuration={key[0]}, Constant Factor={key[1]}, Interaction Strength={key[2]}: {value} failures")

    # Clustering Analysis to Group Failures
    print("\n=== Clustering Analysis of Failures ===")
    if len(df) > 10:
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
