import numpy as np
import gudhi as gd
from scipy.spatial import ConvexHull
import random
import gc
import statistics

# Function to verify if a point set forms a convex position in d-dimensions
def verify_convex_position(points):
    try:
        ConvexHull(points)
        return True
    except:
        return False

# Simulated Annealing Function for Convex Positioning
def simulated_annealing(points, num_iterations=1000, initial_temperature=10.0, cooling_rate=0.95):
    current_points = np.copy(points).astype(np.float32)
    current_convex_status = verify_convex_position(current_points)
    current_score = float('inf') if not current_convex_status else 0
    temperature = initial_temperature

    for iteration in range(num_iterations):
        # Make a small random adjustment to a random point
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=0.1, size=current_points.shape[1]).astype(np.float32)
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment

        # Check if the new configuration forms a convex hull
        new_convex_status = verify_convex_position(new_points)
        new_score = float('inf') if not new_convex_status else 0

        # Calculate the acceptance probability
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

        # If a convex configuration is found, break early
        if current_score == 0:
            break

    return current_points, current_score

# Function to compute persistent homology of point clouds
def compute_persistent_homology(points, max_dimension=2):
    rips_complex = gd.RipsComplex(points=points, max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    # Get persistence diagram
    persistence = simplex_tree.persistence()
    return persistence

# Comprehensive Empirical Testing for Dimensions 3 to 20
# Parameters for testing
max_dimension = 20
comprehensive_num_trials = 5  # Reduced to ensure efficiency across larger dimension range
comprehensive_num_iterations = 1000
comprehensive_initial_temperature = 10.0
comprehensive_cooling_rate = 0.95
statistics_summary = {}

# Run testing for dimensions d = 3 to d = 20
for dimension in range(3, max_dimension + 1):
    success_trials = 0
    num_points = 2 * dimension + 1
    dimension_persistence_summary = []

    for trial in range(comprehensive_num_trials):
        # Generate random points in the current dimensional space
        points = np.random.uniform(low=0, high=1, size=(num_points, dimension)).astype(np.float32)

        # Apply simulated annealing to get a convex configuration
        optimized_points, final_score = simulated_annealing(
            points,
            num_iterations=comprehensive_num_iterations,
            initial_temperature=comprehensive_initial_temperature,
            cooling_rate=comprehensive_cooling_rate
        )

        # Verify if the optimized points form a convex hull
        is_convex_position = verify_convex_position(optimized_points)
        if is_convex_position:
            success_trials += 1

        # Compute persistent homology of the optimized point cloud
        persistence = compute_persistent_homology(optimized_points, max_dimension=2)

        # Extract number of components and their persistence intervals
        num_components = sum(1 for interval in persistence if interval[0] == 0)
        dimension_persistence_summary.append(num_components)

        # Clean up memory
        del points, optimized_points
        gc.collect()

    # Calculate statistics for the current dimension
    mean_components = statistics.mean(dimension_persistence_summary)
    stdev_components = statistics.stdev(dimension_persistence_summary) if len(dimension_persistence_summary) > 1 else 0

    statistics_summary[dimension] = {
        "success_rate": success_trials / comprehensive_num_trials,
        "mean_num_components": mean_components,
        "stdev_num_components": stdev_components
    }

    # Print summary for each dimension
    print(f"Dimension {dimension}: Success Rate = {statistics_summary[dimension]['success_rate']:.2f}, "
          f"Mean # of Components = {mean_components:.2f}, "
          f"StdDev # of Components = {stdev_components:.2f}")

# Print final summary of all dimensions
for dim, summary in statistics_summary.items():
    print(f"Dimension {dim}: Success Rate = {summary['success_rate']:.2f}, "
          f"Mean # of Components = {summary['mean_num_components']:.2f}, "
          f"StdDev # of Components = {summary['stdev_num_components']:.2f}")
