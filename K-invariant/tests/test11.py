import gudhi as gd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tqdm import tqdm

# Batch size for each parameter set
batch_size = 10000

# Adaptive model to suggest parameters based on failure data
adaptive_model = RandomForestRegressor(n_estimators=10, random_state=42)
failure_history = []

# Function to calculate enhanced K_M with added structural terms
def calculate_enhanced_K_M(c_values, x_values, c_0, betti_numbers, interaction_matrix, exponential=False):
    linear_term = np.dot(c_values, x_values)
    betti_interaction_term = np.sum(interaction_matrix)

    if exponential:
        exponential_term = np.exp(np.sum(betti_numbers))
        return linear_term + betti_interaction_term + c_0 + exponential_term
    else:
        return linear_term + betti_interaction_term + c_0

# Extended Parameter Exploration with Finer Granularity
parameter_batches = [
    {"constant_factor": f, "interaction_strength": s}
    for f in np.arange(1.5, 3.1, 0.05)  # Constant factors from 1.5 to 3.0, finer increment
    for s in np.arange(0.1, 1.1, 0.05)  # Interaction strengths from 0.1 to 1.0, finer increment
]

# Loop through each parameter set and run tests for different configurations
for param_set in parameter_batches:
    constant_factor = param_set["constant_factor"]
    interaction_strength = param_set["interaction_strength"]

    # Enhanced K_M with additional structural components
    print(f"\nRunning Enhanced K_M with Parameters: Constant Factor={constant_factor}, Interaction Strength={interaction_strength}")
    successful_bounds = 0
    batch_failures = []

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

        # Calculate enhanced K_M with even more structural terms
        c_values = np.random.uniform(-5, 5, num_vertices)
        x_values = np.random.uniform(-10, 15, num_vertices)
        c_0 = np.random.uniform(-5, 5)

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
            batch_failures.append({
                'K_M': K_M,
                'Complexity': complexity,
                'Max_Dimension': max_dimension,
                'Vertices': num_vertices,
                'Constant Factor': constant_factor,
                'Interaction Strength': interaction_strength,
                'Betti Numbers': betti_numbers
            })

    # Store failure data
    failure_history.extend(batch_failures)

    # Summary of Enhanced K_M Batch Results
    print("\n=== Enhanced K_M Batch Simulation Results ===")
    print(f"Total Tests: {batch_size}")
    print(f"Successful Bound Checks: {successful_bounds}")
    print(f"Failure Cases: {batch_size - successful_bounds}")
    print(f"Success Rate: {successful_bounds / batch_size * 100:.2f}%")

# Use failure data to train adaptive model
if len(failure_history) > 0:
    df_failures = pd.DataFrame(failure_history)
    X_train = df_failures[['Complexity', 'Max_Dimension', 'Vertices', 'Constant Factor', 'Interaction Strength']]
    y_train = df_failures['K_M']
    adaptive_model.fit(X_train, y_train)

    print("\n=== Adaptive Model Trained ===")
else:
    print("\nSUCCESS: No failures across all parameter batches.")

# Clustering Analysis of Failures (if failures still exist)
if len(failure_history) > 0:
    print("\n=== Clustering Analysis of Failures ===")
    df = pd.DataFrame(failure_history)
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
