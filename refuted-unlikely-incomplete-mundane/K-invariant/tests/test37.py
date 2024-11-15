import numpy as np
from gudhi import RipsComplex
import pandas as pd
from tqdm import tqdm

def sample_n_sphere(n, num_points, radius=1.0, noise_level=0.0):
    """Sample points from n-sphere with optional Gaussian noise"""
    points = np.random.normal(0, 1, (num_points, n+1))
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis] * radius
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, points.shape)
        points += noise
    return points

def compute_k_invariant(persistence_diagram, dimension, num_points):
    if not persistence_diagram:
        return 0.0

    lifetimes = []
    for dim, (birth, death) in persistence_diagram:
        if death != float('inf'):
            lifetimes.append(death - birth)

    if not lifetimes:
        return 0.0

    lifetimes = np.array(lifetimes)
    base_term = np.sum(lifetimes**2)
    sin_term = np.sum(np.sin(lifetimes * np.pi))
    # Adjusted scaling factor for higher dimensions
    dim_factor = np.exp(0.15 * dimension)
    point_factor = np.log(1 + num_points)

    return dim_factor * (base_term + sin_term + point_factor)

def run_comprehensive_test(dimensions=range(2, 11),
                         point_counts=[50, 100, 200],
                         noise_levels=[0.0, 0.05, 0.1],
                         trials=3):
    results = []

    total_tests = len(dimensions) * len(point_counts) * len(noise_levels) * trials
    test_counter = 0

    for dim in dimensions:
        for num_points in point_counts:
            for noise in noise_levels:
                for trial in range(trials):
                    test_counter += 1
                    print(f"\nTest {test_counter}/{total_tests}")
                    print(f"Dimension: {dim}, Points: {num_points}, Noise: {noise:.2f}, Trial: {trial+1}/{trials}")

                    # Sample points
                    points = sample_n_sphere(dim, num_points, noise_level=noise)

                    # Compute persistence
                    rips = RipsComplex(points=points, max_edge_length=2.0)
                    st = rips.create_simplex_tree(max_dimension=dim+1)
                    persistence = st.persistence()

                    # Compute invariant
                    k_value = compute_k_invariant(persistence, dim, num_points)
                    true_complexity = 2**dim

                    results.append({
                        'dimension': dim,
                        'num_points': num_points,
                        'noise_level': noise,
                        'k_invariant': k_value,
                        'true_complexity': true_complexity,
                        'bound_satisfied': k_value >= true_complexity,
                        'num_features': len(persistence)
                    })

                    print(f"K-invariant: {k_value:.2f}, True complexity: {true_complexity}")
                    print(f"Number of features: {len(persistence)}")

    return pd.DataFrame(results)

# Run comprehensive test
print("Starting comprehensive test...")
results_df = run_comprehensive_test()

# Analysis of results
print("\nOverall Summary by Dimension:")
dimension_summary = results_df.groupby('dimension').agg({
    'k_invariant': ['mean', 'std'],
    'true_complexity': 'mean',
    'bound_satisfied': 'mean',
    'num_features': 'mean'
}).round(4)
print(dimension_summary)

print("\nEffect of Noise:")
noise_summary = results_df.groupby(['dimension', 'noise_level']).agg({
    'k_invariant': ['mean', 'std'],
    'bound_satisfied': 'mean'
}).round(4)
print(noise_summary)

print("\nEffect of Sample Size:")
size_summary = results_df.groupby(['dimension', 'num_points']).agg({
    'k_invariant': ['mean', 'std'],
    'bound_satisfied': 'mean'
}).round(4)
print(size_summary)

# Additional Analysis
print("\nValidity Checks:")
print(f"1. Always bounds complexity: {results_df['bound_satisfied'].all()}")
print(f"2. Grows with dimension: {results_df.groupby('dimension')['k_invariant'].mean().is_monotonic_increasing}")

# Correlation analysis
for noise in results_df['noise_level'].unique():
    noise_data = results_df[results_df['noise_level'] == noise]
    correlation = np.corrcoef(
        noise_data.groupby('dimension')['k_invariant'].mean(),
        noise_data.groupby('dimension')['true_complexity'].mean()
    )[0,1]
    print(f"3. Correlation with theoretical complexity (noise={noise}): {correlation:.4f}")

# Save results to CSV
results_df.to_csv('k_invariant_test_results.csv', index=False)
print("\nDetailed results saved to 'k_invariant_test_results.csv'")
