import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================================
# PART 1: Updated Helper Functions
# ===========================================

def generate_sphere_points(radius, num_points, dim, uniform=True):
    """
    Generates points on the surface of a sphere with optional uniform or non-uniform distribution.
    Parameters:
    - radius (float): The radius of the sphere.
    - num_points (int): The number of points to generate.
    - dim (int): The dimension of the sphere.
    - uniform (bool): Whether to use uniform sampling or non-uniform sampling.
    """
    if uniform:
        # Uniform sampling by normalizing points drawn from a normal distribution
        points = np.random.randn(num_points, dim + 1)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    else:
        # Non-uniform sampling: bias the distribution by introducing a weighted factor
        points = np.random.randn(num_points, dim + 1)
        points = points * (1 + np.random.uniform(0.5, 1.5, size=(num_points, dim + 1)))  # Introduce a bias
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    points *= radius
    return points

def add_noise_to_points(points, noise_level):
    """
    Adds Gaussian noise to points on the sphere.
    Parameters:
    - points (ndarray): The array of points to add noise to.
    - noise_level (float): Standard deviation of the noise relative to the radius.
    """
    noise = np.random.normal(scale=noise_level, size=points.shape)
    noisy_points = points + noise
    return noisy_points

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
# PART 2: Parallel Processing for Different Sampling and Noise Levels
# ===========================================

def analyze_single_iteration(radius, num_points, dimension, uniform=True, noise_level=0.0):
    # Generate sphere points (either uniform or non-uniform)
    points = generate_sphere_points(radius, num_points, dimension, uniform=uniform)

    # Apply stepwise noise if noise_level > 0
    if noise_level > 0:
        points = add_noise_to_points(points, noise_level)

    # Compute persistent homology
    persistence_intervals_0, persistence_intervals_1 = compute_persistent_homology(points)

    # Calculate K_invariant
    complexity = len(persistence_intervals_0) + len(persistence_intervals_1)
    invariant_value = k_invariant([persistence_intervals_0, persistence_intervals_1], num_points, dimension)

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
# PART 3: Running the Analysis with 500 Points, Different Noise Levels, and Mixed Sampling
# ===========================================

if __name__ == '__main__':
    radius = 1.0
    num_points = 500  # Number of points set to 500
    iterations = 50  # Number of repetitions per configuration for statistical stability
    noise_levels = [0.0, 0.05, 0.1, 0.15]  # Stepwise noise levels to test resilience

    # Analyzing dimensions from 9 to 15 for increased rigor
    max_dimension = 15
    all_results = []

    for dim in range(9, max_dimension + 1):
        sphere_analysis_df = analyze_spheres_in_parallel(radius, num_points, dim, iterations, noise_levels)
        success_rate = sphere_analysis_df["bound_check"].mean() * 100
        print(f"Success Rate for S^{dim}: {success_rate:.2f}%")
        all_results.append(sphere_analysis_df)

    # Combine all results into a single DataFrame for easier analysis
    all_spheres_analysis_df = pd.concat(all_results, ignore_index=True)

    # Optionally, save to a CSV file for later inspection
    all_spheres_analysis_df.to_csv("spheres_analysis_mixed_sampling_with_noise.csv", index=False)
    print("Analysis complete and results saved.")
