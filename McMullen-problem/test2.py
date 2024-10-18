import numpy as np
from scipy.spatial import ConvexHull
import random
import gc

# Function to verify if a point set forms a convex position in d-dimensions
def verify_convex_position(points):
    try:
        ConvexHull(points)
        return True
    except:
        return False

# Simulated Annealing Function for Convex Positioning
def simulated_annealing(points, num_iterations=500, initial_temperature=10.0, cooling_rate=0.95):
    current_points = np.copy(points).astype(np.float32)
    current_convex_status = verify_convex_position(current_points)
    current_score = float('inf') if not current_convex_status else 0

    temperature = initial_temperature

    for iteration in range(num_iterations):
        # Make a small random adjustment to a random point
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=0.1, size=current_points.shape[1]).astype(np.float32)
        new_points = current_points.copy()
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

# Hyper-Rigor Testing for Dimensions 3 to 16
def run_hyper_rigor_test(max_dimension=20, num_trials=10, num_iterations=500):
    results = {}
    for dimension in range(3, max_dimension + 1):
        results[dimension] = []
        num_points = 2 * dimension + 1

        for trial in range(num_trials):
            # Generate random points in the current dimensional space
            points = np.random.uniform(low=0, high=1, size=(num_points, dimension)).astype(np.float32)

            # Apply simulated annealing to the point set with rigorous parameters
            optimized_points, final_score = simulated_annealing(
                points,
                num_iterations=num_iterations,
                initial_temperature=10.0,
                cooling_rate=0.95
            )

            # Verify if the optimized points form a convex hull in d-dimensions
            is_convex_position = verify_convex_position(optimized_points)
            results[dimension].append(is_convex_position)

            # Clean up memory
            del points, optimized_points
            gc.collect()

        # Print result for each dimension for tracking purposes
        print(f"Dimension {dimension}, Convex Position Achieved (in {num_trials} trials): {results[dimension]}")

    # Summary of rigorous results from dimensions 3 to max_dimension
    print("Rigorous Testing Summary:", results)

if __name__ == "__main__":
    run_hyper_rigor_test(max_dimension=20, num_trials=10, num_iterations=500)
