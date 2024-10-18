import numpy as np
from scipy.spatial import ConvexHull
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
import logging
import time
import math
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(
    filename='mcullen_verification.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Enable logging to console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def verify_convex_position(points):
    """
    Verify if the given set of points forms a convex position in d-dimensional space.

    Args:
        points (np.ndarray): An array of shape (n_points, d) representing the point set.

    Returns:
        bool: True if all points are vertices of the convex hull, False otherwise.
    """
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception as e:
        logging.debug(f"ConvexHull computation failed: {e}")
        return False

def calculate_convex_hull_volume(points):
    """
    Calculate the volume of the convex hull of the given points.

    Args:
        points (np.ndarray): An array of shape (n_points, d) representing the point set.

    Returns:
        float: Volume of the convex hull. Returns 0 if convex hull cannot be computed.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def simulated_annealing(points, num_iterations=1000, initial_temperature=10.0, cooling_rate=0.95, adjustment_scale=0.1):
    """
    Perform simulated annealing to transform a set of points into a convex configuration.

    Args:
        points (np.ndarray): Initial set of points of shape (n_points, d).
        num_iterations (int): Number of iterations for the annealing process.
        initial_temperature (float): Starting temperature for the annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        adjustment_scale (float): Scale of the random adjustments to points.

    Returns:
        dict: A dictionary containing detailed results of the optimization.
    """
    current_points = np.copy(points)
    d = current_points.shape[1]
    current_score = 0 if verify_convex_position(current_points) else 1

    temperature = initial_temperature
    history = {
        'temperature': [],
        'score': [],
        'accepted_moves': 0,
        'rejected_moves': 0,
        'iterations': 0
    }

    for iteration in range(1, num_iterations + 1):
        # Record history
        history['temperature'].append(temperature)
        history['score'].append(current_score)

        # Make a small random adjustment to a random point
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=adjustment_scale, size=current_points.shape[1])
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment

        # Check convexity
        new_score = 0 if verify_convex_position(new_points) else 1

        # Calculate acceptance probability
        acceptance_probability = math.exp((current_score - new_score) / temperature) if new_score >= current_score else 1.0

        # Decide whether to accept the new configuration
        if random.random() < acceptance_probability:
            current_points = new_points
            current_score = new_score
            history['accepted_moves'] += 1
        else:
            history['rejected_moves'] += 1

        # Cool down the temperature
        temperature *= cooling_rate
        history['iterations'] = iteration

        # If a convex configuration is found, break early
        if current_score == 0:
            break

    return current_points, current_score, history

def run_trial(dimension, num_trials=10, num_iterations=1000):
    """
    Run multiple trials of simulated annealing for a given dimension.

    Args:
        dimension (int): The dimension of the affine space.
        num_trials (int): Number of independent trials to run.
        num_iterations (int): Number of iterations for each annealing process.

    Returns:
        list: A list of dictionaries containing the trial results.
    """
    results = []
    num_points = 2 * dimension + 1

    for trial in range(num_trials):
        points = np.random.uniform(low=0, high=1, size=(num_points, dimension))

        optimized_points, final_score, history = simulated_annealing(
            points, num_iterations=num_iterations
        )

        is_convex = verify_convex_position(optimized_points)
        result = {
            'is_convex': is_convex,
            'optimized_points': optimized_points.tolist(),
            'final_score': final_score,
            'history': history
        }
        results.append(result)

        gc.collect()

    return results

def run_mcullen_verification(max_dimension=20, num_trials=10, num_iterations=1000):
    """
    Run the McMullen Problem verification across multiple dimensions.

    Args:
        max_dimension (int): The maximum dimension to test.
        num_trials (int): Number of independent trials per dimension.
        num_iterations (int): Number of iterations for each annealing process.

    Returns:
        dict: A dictionary with dimensions as keys and lists of trial results as values.
    """
    results = {}
    start_time = time.time()
    logging.info("Starting McMullen Verification")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_trial, dimension, num_trials, num_iterations): dimension
            for dimension in range(3, max_dimension + 1)
        }
        for future in as_completed(futures):
            dimension = futures[future]
            try:
                trial_results = future.result()
                results[dimension] = trial_results
                success_rate = sum(trial['is_convex'] for trial in trial_results) / len(trial_results)
                logging.info(f"Dimension {dimension}: Success Rate = {success_rate:.2f}")
            except Exception as e:
                logging.error(f"Dimension {dimension} failed: {e}")

    total_elapsed_time = time.time() - start_time
    logging.info(f"All trials completed in {total_elapsed_time:.2f} seconds.")
    return results

def analyze_results(results, output_dir='results_analysis'):
    """
    Analyze the results of the trials and generate summary statistics and visualizations.

    Args:
        results (dict): A nested dictionary containing all trial results organized by dimension.
        output_dir (str): Directory where analysis results will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary = {}

    for dimension, trials in results.items():
        successful_trials = sum(1 for trial in trials if trial['is_convex'])
        success_rate = successful_trials / len(trials)
        iterations = [trial['history']['iterations'] for trial in trials]

        summary[dimension] = {
            'success_rate': success_rate,
            'average_iterations': np.mean(iterations),
            'std_iterations': np.std(iterations)
        }

        # Plot Success Rate
        plt.figure(figsize=(6, 4))
        plt.bar([dimension], [success_rate], color='green' if success_rate == 1.0 else 'orange')
        plt.ylim(0, 1)
        plt.xlabel('Dimension')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate for Dimension {dimension}')
        plt.savefig(os.path.join(output_dir, f'Success_Rate_Dim_{dimension}.png'))
        plt.close()

    # Save summary statistics
    summary_file = os.path.join(output_dir, 'summary_statistics.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    # Parameters
    max_dimension = 20
    num_trials = 10
    num_iterations = 1000

    # Run the verification
    results = run_mcullen_verification(max_dimension, num_trials, num_iterations)
    analyze_results(results)
    print("Final results:", results)
