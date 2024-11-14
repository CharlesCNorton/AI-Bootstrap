# Write your code here :-)
import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================================
# PART 1: Updated Helper Functions
# ===========================================

def generate_sphere_points(radius, num_points, dim, uniform=True):
    if uniform:
        points = np.random.randn(num_points, dim + 1)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    else:
        points = np.random.randn(num_points, dim + 1)
        points = points * (1 + np.random.uniform(0.5, 1.5, size=(num_points, dim + 1)))  # Introduce a bias
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    points *= radius
    return points

def add_noise_to_points(points, noise_level):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def compute_persistent_homology(points):
    # Compute persistent homology and return intervals for dimensions 0, 1, and 2
    result = ripser(points)
    dgms = result['dgms']
    return dgms[0], dgms[1], dgms[2] if len(dgms) > 2 else []

def k_invariant(persistence_intervals, num_points, dimension, points):
    # Base invariant: Sum of log-transformed lifetimes squared (with clipping)
    base_invariant = 0
    for dim_intervals in persistence_intervals:
        for interval in dim_intervals:
            if interval[1] < np.inf:  # Ensure finite lifetime
                lifetime = interval[1] - interval[0]
                base_invariant += np.clip(np.log1p(lifetime) ** 2, 0, 1e6)

    # Cross-term to account for interactions across intervals in the same dimension
    cross_term = 0
    for dim_intervals in persistence_intervals:
        if len(dim_intervals) > 1:
            for i in range(len(dim_intervals) - 1):
                interval_1 = dim_intervals[i]
                interval_2 = dim_intervals[i + 1]
                if interval_1[1] < np.inf and interval_2[1] < np.inf:
                    cross_term += np.clip(abs((interval_1[1] - interval_1[0]) * (interval_2[1] - interval_2[0])), 0, 1e6)

    # Logarithmic scaling term for minimum complexity bound, enhanced to scale with sphere dimension
    log_term = np.log1p(num_points * dimension)

    # Fourier series to capture periodic homotopical behaviors in stable homotopy groups
    periodic_term = 0
    for dim_intervals in persistence_intervals:
        for interval in dim_intervals:
            if interval[1] < np.inf:  # Only consider finite intervals
                periodic_term += np.clip(np.sin((interval[1] - interval[0]) * np.pi / 2), -1e3, 1e3)

    # Adaptive scaling factor for dimensional adjustments
    adaptive_scaling = 1 + (dimension ** 0.5) * 0.2 + np.exp(0.02 * dimension)

    # Geometric feature: Average distance between points
    avg_distance = np.mean([np.linalg.norm(p1 - p2) for i, p1 in enumerate(points) for j, p2 in enumerate(points) if i < j])

    # Combining all components for the enhanced K_invariant (clipped to avoid numerical overflow)
    refined_invariant = adaptive_scaling * (base_invariant + cross_term + log_term + periodic_term + avg_distance)
    refined_invariant = np.clip(refined_invariant, 0, 1e12)  # Clip final value to avoid runaway values

    return refined_invariant

# ===========================================
# PART 2: Parallel Processing for Different Sampling and Noise Levels
# ===========================================

def analyze_single_iteration(radius, num_points, dimension, uniform=True, noise_level=0.0):
    points = generate_sphere_points(radius, num_points, dimension, uniform=uniform)

    if noise_level > 0:
        points = add_noise_to_points(points, noise_level)

    persistence_intervals_0, persistence_intervals_1, persistence_intervals_2 = compute_persistent_homology(points)

    complexity = len(persistence_intervals_0) + len(persistence_intervals_1) + len(persistence_intervals_2)
    invariant_value = k_invariant([persistence_intervals_0, persistence_intervals_1, persistence_intervals_2], num_points, dimension, points)

    bound_check = invariant_value >= complexity
    return {
        "dimension": dimension,
        "num_points": num_points,
        "complexity": complexity,
        "K_invariant": invariant_value,
        "bound_check": bound_check,
        "uniform": uniform,
        "noise_level": noise_level
    }

def analyze_spheres_in_parallel(radius, num_points, dimension, iterations=100, noise_levels=[0.0, 0.05, 0.1]):
    results = []

    # Using max parallelism with ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = []

        # Submit iterations for both uniform and non-uniform sampling, with different noise levels
        for noise_level in noise_levels:
            for _ in range(iterations):
                # Uniform sampling
                futures.append(executor.submit(analyze_single_iteration, radius, num_points, dimension, True, noise_level))
                # Non-uniform sampling
                futures.append(executor.submit(analyze_single_iteration, radius, num_points, dimension, False, noise_level))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"S^{dimension}"):
            results.append(future.result())

    df = pd.DataFrame(results)
    return df

# ===========================================
# PART 3: Running the Analysis with Cross-Dimensional Checks and Statistical Summaries
# ===========================================

if __name__ == '__main__':
    radius = 1.0
    num_points = 1000
    iterations = 50
    noise_levels = [0.0, 0.05, 0.1, 0.15]

    max_dimension = 25
    all_results = []

    for dim in range(9, max_dimension + 1):
        # Perform analysis for the given dimension
        sphere_analysis_df = analyze_spheres_in_parallel(radius, num_points, dim, iterations, noise_levels)

        # Statistical Summary for each dimension
        mean_complexity = sphere_analysis_df['complexity'].mean()
        std_complexity = sphere_analysis_df['complexity'].std()
        min_complexity = sphere_analysis_df['complexity'].min()
        max_complexity = sphere_analysis_df['complexity'].max()

        mean_k_invariant = sphere_analysis_df['K_invariant'].mean()
        std_k_invariant = sphere_analysis_df['K_invariant'].std()
        min_k_invariant = sphere_analysis_df['K_invariant'].min()
        max_k_invariant = sphere_analysis_df['K_invariant'].max()

        success_rate = sphere_analysis_df["bound_check"].mean() * 100

        # Printing the summary
        print(f"--- Dimension S^{dim} ---")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Complexity - Mean: {mean_complexity:.2f}, Std: {std_complexity:.2f}, Min: {min_complexity}, Max: {max_complexity}")
        print(f"K_invariant - Mean: {mean_k_invariant:.2f}, Std: {std_k_invariant:.2f}, Min: {min_k_invariant}, Max: {max_k_invariant}")
        print()

        # Store the results for cross-dimensional analysis
        all_results.append(sphere_analysis_df)

    # Combine all results into a single DataFrame for easier analysis
    all_spheres_analysis_df = pd.concat(all_results, ignore_index=True)
    all_spheres_analysis_df.to_csv("spheres_analysis_mixed_sampling_with_noise_improved.csv", index=False)
    print("Analysis complete and results saved.")

    # Cross-Dimensional Analysis Summary
    cross_dim_summary = all_spheres_analysis_df.groupby('dimension').agg({
        'complexity': ['mean', 'std'],
        'K_invariant': ['mean', 'std'],
        'bound_check': 'mean'
    }).reset_index()

    # Printing Cross-Dimensional Checks
    print("--- Cross-Dimensional Analysis Summary ---")
    for _, row in cross_dim_summary.iterrows():
        dimension = int(row['dimension'])
        mean_complexity = row[('complexity', 'mean')]
        std_complexity = row[('complexity', 'std')]
        mean_k_invariant = row[('K_invariant', 'mean')]
        std_k_invariant = row[('K_invariant', 'std')]
        success_rate = row[('bound_check', 'mean')] * 100

        print(f"Dimension S^{dimension}:")
        print(f"  Mean Complexity: {mean_complexity:.2f}, Std Complexity: {std_complexity:.2f}")
        print(f"  Mean K_invariant: {mean_k_invariant:.2f}, Std K_invariant: {std_k_invariant:.2f}")
        print(f"  Success Rate: {success_rate:.2f}%")
        print()
