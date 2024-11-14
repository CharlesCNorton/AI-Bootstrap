import numpy as np
from gudhi import RipsComplex
import pandas as pd
from tqdm import tqdm

def sample_n_sphere(n, num_points, radius=1.0):
    points = np.random.normal(0, 1, (num_points, n+1))
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis] * radius
    return points

def compute_k_invariant(persistence_diagram, dimension, num_points):
    if not persistence_diagram:
        return 0.0

    # Print persistence diagram for debugging
    print(f"Persistence diagram for dimension {dimension}:", persistence_diagram[:5])

    lifetimes = []
    for dim, (birth, death) in persistence_diagram:
        if death != float('inf'):
            lifetimes.append(death - birth)

    if not lifetimes:
        return 0.0

    lifetimes = np.array(lifetimes)
    base_term = np.sum(lifetimes**2)
    sin_term = np.sum(np.sin(lifetimes * np.pi))
    dim_factor = np.exp(0.1 * dimension)
    point_factor = np.log(1 + num_points)

    return dim_factor * (base_term + sin_term + point_factor)

def run_test(dimensions=[2,3,4,5], num_points=50, trials=3):
    results = []

    for dim in dimensions:
        print(f"\nProcessing dimension {dim}")
        for trial in range(trials):
            print(f"Trial {trial + 1}/{trials}")

            # Sample points
            points = sample_n_sphere(dim, num_points)
            print(f"Generated {len(points)} points in dimension {dim}")

            # Compute persistence
            rips = RipsComplex(points=points, max_edge_length=2.0)
            st = rips.create_simplex_tree(max_dimension=dim+1)
            persistence = st.persistence()
            print(f"Computed persistence with {len(persistence)} features")

            # Compute invariant
            k_value = compute_k_invariant(persistence, dim, num_points)
            true_complexity = 2**dim  # Fixed: using dim instead of dimension

            results.append({
                'dimension': dim,
                'k_invariant': k_value,
                'true_complexity': true_complexity,
                'bound_satisfied': k_value >= true_complexity
            })

            print(f"K-invariant: {k_value:.2f}, True complexity: {true_complexity}")

    return pd.DataFrame(results)

# Run with minimal parameters for testing
print("Starting test with minimal parameters...")
results_df = run_test()

print("\nFinal Results:")
print(results_df)

# Basic statistics
print("\nSummary by dimension:")
summary = results_df.groupby('dimension').agg({
    'k_invariant': ['mean', 'std'],
    'true_complexity': 'mean',
    'bound_satisfied': 'mean'
}).round(4)
print(summary)
