import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================================
# PART 1: Updated Helper Functions
# ===========================================

def generate_sphere_points(radius, num_points, dim):
    points = np.random.randn(num_points, dim + 1)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= radius
    return points

def compute_persistent_homology(points):
    # Use Ripser to compute persistent homology of the point cloud
    result = ripser(points)
    dgms = result['dgms']

    # Return persistence intervals for dimensions 0 and 1 (Ripser returns intervals for all dimensions)
    return dgms[0], dgms[1] if len(dgms) > 1 else []

def k_invariant(persistence_intervals, num_points, dimension):
    base_invariant = sum([(interval[1] - interval[0]) ** 2 for dim_intervals in persistence_intervals for interval in dim_intervals if interval[1] < np.inf])

    cross_term = 0
    if len(persistence_intervals[0]) > 1 and len(persistence_intervals[1]) > 1:
        for interval_0, interval_1 in zip(persistence_intervals[0], persistence_intervals[1]):
            cross_term += abs((interval_0[1] - interval_0[0]) * (interval_1[1] - interval_1[0]))

    log_term = np.log1p(num_points * dimension)
    periodic_term = sum(np.sin((interval[1] - interval[0]) * np.pi / 2) for dim_intervals in persistence_intervals for interval in dim_intervals if interval[1] < np.inf)
    adaptive_scaling = 1 + (dimension ** 0.5) * 0.1

    refined_invariant = adaptive_scaling * (base_invariant + cross_term + log_term + periodic_term)

    return refined_invariant

# ===========================================
# PART 2: Parallel Processing for Dimensions 9 and 10
# ===========================================

def analyze_single_iteration(radius, num_points, dimension):
    points = generate_sphere_points(radius, num_points, dimension)
    persistence_intervals_0, persistence_intervals_1 = compute_persistent_homology(points)

    complexity = len(persistence_intervals_0) + len(persistence_intervals_1)
    invariant_value = k_invariant([persistence_intervals_0, persistence_intervals_1], num_points, dimension)

    bound_check = invariant_value >= complexity
    return {
        "dimension": dimension,
        "num_points": num_points,
        "complexity": complexity,
        "K_invariant": invariant_value,
        "bound_check": bound_check
    }

def analyze_spheres_in_parallel(radius, num_points, dimension, iterations=100):
    results = []

    # Using max parallelism with ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_single_iteration, radius, num_points, dimension) for _ in range(iterations)]

        for future in tqdm(as_completed(futures), total=iterations, desc=f"S^{dimension}"):
            results.append(future.result())

    df = pd.DataFrame(results)
    return df

# ===========================================
# PART 3: Running the Analysis for Dimensions 9 and 10
# ===========================================

if __name__ == '__main__':
    radius = 1.0
    num_points = 100  # Increase for better approximations
    iterations = 100  # Number of repetitions per dimension for statistical stability

    # Run analysis for dimensions 9 and 10
    all_spheres_analysis_df_9 = analyze_spheres_in_parallel(radius, num_points, 9, iterations)
    all_spheres_analysis_df_10 = analyze_spheres_in_parallel(radius, num_points, 10, iterations)

    # Calculate success rate
    success_rate_9 = all_spheres_analysis_df_9["bound_check"].mean() * 100
    success_rate_10 = all_spheres_analysis_df_10["bound_check"].mean() * 100

    print(f"Success Rate for S^9: {success_rate_9:.2f}%")
    print(f"Success Rate for S^10: {success_rate_10:.2f}%")
