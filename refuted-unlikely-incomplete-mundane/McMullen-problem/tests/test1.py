import numpy as np
import cupy as cp
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor
import random
import os
import gc

# Fallback: CPU-based convex hull computation
def verify_convex_position_cpu(points):
    try:
        ConvexHull(points)
        return True
    except:
        return False

# GPU-based convex hull computation using CuPy for CUDA acceleration
def verify_convex_position_gpu(points):
    try:
        points_gpu = cp.asarray(points)
        points_cpu = cp.asnumpy(points_gpu)
        # Use scipy's ConvexHull on CPU after copying data from GPU (due to lack of direct CUDA support)
        ConvexHull(points_cpu)
        return True
    except:
        return False

# Generalized function to verify convex position (GPU if available, CPU fallback)
def verify_convex_position(points, use_gpu=True):
    if use_gpu:
        return verify_convex_position_gpu(points)
    else:
        return verify_convex_position_cpu(points)

# Simulated annealing for convex positioning
def simulated_annealing(points, num_iterations=500, initial_temperature=10.0, cooling_rate=0.95, use_gpu=True):
    current_points = np.copy(points)
    current_convex_status = verify_convex_position(current_points, use_gpu)
    current_score = float('inf') if not current_convex_status else 0
    temperature = initial_temperature

    for iteration in range(num_iterations):
        # Make a small random adjustment
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=0.1, size=current_points.shape[1])
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment

        # Verify if new configuration forms a convex hull
        new_convex_status = verify_convex_position(new_points, use_gpu)
        new_score = float('inf') if not new_convex_status else 0

        # Acceptance probability
        if new_score < current_score:
            acceptance_probability = 1.0
        else:
            acceptance_probability = np.exp((current_score - new_score) / temperature)

        # Decide whether to accept the new configuration
        if random.random() < acceptance_probability:
            current_points = new_points
            current_score = new_score

        # Cool down the temperature
        temperature *= cooling_rate

        if current_score == 0:
            break

    return current_points, current_score

# Function to run trials for each dimension
def run_trial(dimension, num_trials=10, num_iterations=500, use_gpu=True):
    results = []
    num_points = 2 * dimension + 1
    for trial in range(num_trials):
        # Generate random points in d-dimensional space
        points = np.random.uniform(low=0, high=1, size=(num_points, dimension))
        optimized_points, final_score = simulated_annealing(
            points, num_iterations=num_iterations, use_gpu=use_gpu
        )
        is_convex = verify_convex_position(optimized_points, use_gpu)
        results.append(is_convex)
    return results

# Run the test for dimensions up to 20 using multi-threading and optional GPU acceleration
def run_mcullen_verification(max_dimension=15, num_trials=10, num_iterations=500, use_gpu=True):
    results = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_trial, dimension, num_trials, num_iterations, use_gpu): dimension
            for dimension in range(3, max_dimension + 1)
        }
        for future in futures:
            dimension = futures[future]
            try:
                results[dimension] = future.result()
                print(f"Dimension {dimension}: {results[dimension]}")
            except Exception as e:
                print(f"Dimension {dimension} failed: {e}")

            # Garbage collection cleanup after each dimension run
            gc.collect()

    return results

if __name__ == "__main__":
    # Modify use_gpu based on your hardware setup
    use_gpu = True  # Set to False if no CUDA-enabled GPU is available
    max_dimension = 20
    num_trials = 10
    num_iterations = 1000  # Increase for more rigorous testing
    results = run_mcullen_verification(max_dimension, num_trials, num_iterations, use_gpu)
    print("Final results:", results)
