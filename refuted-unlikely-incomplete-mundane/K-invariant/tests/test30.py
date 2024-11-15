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
# PART 2: Parallel Processing for Higher Dimensions with 1000 Points
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
# PART 3: Running the Analysis for Dimensions Up to 25 with 1000 Points
# ===========================================

if __name__ == '__main__':
    radius = 1.0
    num_points = 1000  # Increased the number of points to 1000 to enhance sphere approximation accuracy
    iterations = 100  # Number of repetitions per dimension for statistical stability

    # Analyzing higher dimensions from 9 up to 25
    max_dimension = 25
    all_results = []

    for dim in range(9, max_dimension + 1):
        sphere_analysis_df = analyze_spheres_in_parallel(radius, num_points, dim, iterations)
        success_rate = sphere_analysis_df["bound_check"].mean() * 100
        print(f"Success Rate for S^{dim}: {success_rate:.2f}%")
        all_results.append(sphere_analysis_df)

    # Combine all results into a single DataFrame for easier analysis
    all_spheres_analysis_df = pd.concat(all_results, ignore_index=True)

    # Optionally, save to a CSV file for later inspection
    all_spheres_analysis_df.to_csv("spheres_analysis_up_to_dim_25_with_1000_points.csv", index=False)
    print("Analysis complete and results saved.")
