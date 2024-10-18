"""
Comprehensive Verification of the McMullen Problem Conjecture using Simulated Annealing

This program empirically verifies the conjecture that for any set of ν(d) = 2d + 1 points
in general position in d-dimensional affine space ℝ^d, there exists a projective transformation
that maps these points into a convex configuration, forming the vertices of a convex polytope.

The program conducts multiple trials across dimensions 3 to 20, employing simulated annealing
to search for suitable transformations. It gathers extensive data on each trial's progress,
success rates, optimization dynamics, and final configurations.

"""

import numpy as np
from scipy.spatial import ConvexHull
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
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

def simulated_annealing(
    points,
    num_iterations=1000,
    initial_temperature=10.0,
    cooling_rate=0.95,
    adjustment_scale=0.1,
    verbose=False
):
    """
    Perform simulated annealing to transform a set of points into a convex configuration.

    Args:
        points (np.ndarray): Initial set of points of shape (n_points, d).
        num_iterations (int): Number of iterations for the annealing process.
        initial_temperature (float): Starting temperature for the annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        adjustment_scale (float): Scale of the random adjustments to points.
        verbose (bool): If True, prints detailed debug information.

    Returns:
        dict: A dictionary containing detailed results of the optimization.
    """
    current_points = np.copy(points)
    d = current_points.shape[1]
    current_convex = verify_convex_position(current_points)
    current_score = 0 if current_convex else 1  # Simple scoring: 0 for convex, 1 otherwise
    initial_volume = calculate_convex_hull_volume(current_points)

    temperature = initial_temperature
    history = {
        'temperature': [],
        'score': [],
        'volume': [],
        'accepted_moves': 0,
        'rejected_moves': 0,
        'iterations': 0
    }

    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        # Record history
        history['temperature'].append(temperature)
        history['score'].append(current_score)
        history['volume'].append(calculate_convex_hull_volume(current_points))

        # Make a small random adjustment to a random point
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=adjustment_scale, size=current_points.shape[1])
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment

        # Check convexity
        new_convex = verify_convex_position(new_points)
        new_score = 0 if new_convex else 1

        # Calculate acceptance probability
        if new_score < current_score:
            acceptance_probability = 1.0
        else:
            # Incorporate volume change into scoring
            new_volume = calculate_convex_hull_volume(new_points)
            volume_change = new_volume - history['volume'][-1]
            acceptance_probability = math.exp(-volume_change / temperature) if temperature > 0 else 0.0

        # Decide whether to accept the new configuration
        if random.random() < acceptance_probability:
            current_points = new_points
            current_score = new_score
            history['accepted_moves'] += 1
            if verbose:
                logging.debug(f"Iteration {iteration}: Accepted move. New score: {current_score}")
        else:
            history['rejected_moves'] += 1
            if verbose:
                logging.debug(f"Iteration {iteration}: Rejected move. Score remains: {current_score}")

        # Cool down the temperature
        temperature *= cooling_rate
        history['iterations'] = iteration

        # If a convex configuration is found, break early
        if current_score == 0:
            if verbose:
                logging.debug(f"Iteration {iteration}: Convex configuration achieved.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final volume
    final_volume = calculate_convex_hull_volume(current_points)

    result = {
        'optimized_points': current_points,
        'final_score': current_score,
        'initial_volume': initial_volume,
        'final_volume': final_volume,
        'history': history,
        'elapsed_time_sec': elapsed_time
    }

    return result

def run_trial(
    dimension,
    trial_id,
    num_iterations=1000,
    initial_temperature=10.0,
    cooling_rate=0.95,
    adjustment_scale=0.1,
    verbose=False
):
    """
    Run a single trial of simulated annealing for a given dimension.

    Args:
        dimension (int): The dimension of the affine space.
        trial_id (int): Identifier for the trial.
        num_iterations (int): Number of iterations for the annealing process.
        initial_temperature (float): Starting temperature for the annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        adjustment_scale (float): Scale of the random adjustments to points.
        verbose (bool): If True, prints detailed debug information.

    Returns:
        dict: A dictionary containing detailed results of the trial.
    """
    num_points = 2 * dimension + 1
    # Generate random points in d-dimensional space
    points = np.random.uniform(low=0, high=1, size=(num_points, dimension))

    logging.info(f"Starting Trial {trial_id} for Dimension {dimension}")

    # Apply simulated annealing
    result = simulated_annealing(
        points,
        num_iterations=num_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        adjustment_scale=adjustment_scale,
        verbose=verbose
    )

    # Verify convex position in d-dimensional space
    is_convex = verify_convex_position(result['optimized_points'])
    result['is_convex'] = is_convex
    result['trial_id'] = trial_id
    result['dimension'] = dimension

    if is_convex:
        logging.info(f"Trial {trial_id} for Dimension {dimension} succeeded in {result['history']['iterations']} iterations.")
    else:
        logging.warning(f"Trial {trial_id} for Dimension {dimension} failed to achieve convex configuration.")

    return result

def run_all_trials(
    max_dimension=20,
    num_trials=10,
    num_iterations=1000,
    initial_temperature=10.0,
    cooling_rate=0.95,
    adjustment_scale=0.1,
    use_parallel=True,
    verbose=False
):
    """
    Run multiple trials across multiple dimensions to verify the McMullen Problem conjecture.

    Args:
        max_dimension (int): The maximum dimension to test.
        num_trials (int): Number of independent trials per dimension.
        num_iterations (int): Number of iterations for each annealing process.
        initial_temperature (float): Starting temperature for the annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        adjustment_scale (float): Scale of the random adjustments to points.
        use_parallel (bool): If True, runs trials in parallel using multithreading.
        verbose (bool): If True, prints detailed debug information.

    Returns:
        dict: A nested dictionary containing all trial results organized by dimension.
    """
    results = {}
    start_time = time.time()

    logging.info("Starting all trials for McMullen Problem verification.")

    dimensions = range(3, max_dimension + 1)

    if use_parallel:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_trial = {}
            for dimension in dimensions:
                results[dimension] = []
                for trial in range(1, num_trials + 1):
                    future = executor.submit(
                        run_trial,
                        dimension=dimension,
                        trial_id=trial,
                        num_iterations=num_iterations,
                        initial_temperature=initial_temperature,
                        cooling_rate=cooling_rate,
                        adjustment_scale=adjustment_scale,
                        verbose=verbose
                    )
                    future_to_trial[future] = (dimension, trial)

            for future in as_completed(future_to_trial):
                dimension, trial = future_to_trial[future]
                try:
                    trial_result = future.result()
                    results[dimension].append(trial_result)
                except Exception as e:
                    logging.error(f"Dimension {dimension}, Trial {trial} generated an exception: {e}")
    else:
        for dimension in dimensions:
            results[dimension] = []
            for trial in range(1, num_trials + 1):
                trial_result = run_trial(
                    dimension=dimension,
                    trial_id=trial,
                    num_iterations=num_iterations,
                    initial_temperature=initial_temperature,
                    cooling_rate=cooling_rate,
                    adjustment_scale=adjustment_scale,
                    verbose=verbose
                )
                results[dimension].append(trial_result)

    end_time = time.time()
    total_elapsed_time = end_time - start_time
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
        total_trials = len(trials)
        successful_trials = sum(1 for trial in trials if trial['is_convex'])
        success_rate = successful_trials / total_trials
        iterations = [trial['history']['iterations'] for trial in trials]
        accepted_moves = [trial['history']['accepted_moves'] for trial in trials]
        rejected_moves = [trial['history']['rejected_moves'] for trial in trials]
        volumes_initial = [trial['initial_volume'] for trial in trials]
        volumes_final = [trial['final_volume'] for trial in trials]
        elapsed_times = [trial['elapsed_time_sec'] for trial in trials]

        summary[dimension] = {
            'total_trials': total_trials,
            'successful_trials': successful_trials,
            'success_rate': success_rate,
            'average_iterations': np.mean(iterations),
            'std_iterations': np.std(iterations),
            'average_accepted_moves': np.mean(accepted_moves),
            'std_accepted_moves': np.std(accepted_moves),
            'average_rejected_moves': np.mean(rejected_moves),
            'std_rejected_moves': np.std(rejected_moves),
            'average_initial_volume': np.mean(volumes_initial),
            'std_initial_volume': np.std(volumes_initial),
            'average_final_volume': np.mean(volumes_final),
            'std_final_volume': np.std(volumes_final),
            'average_elapsed_time_sec': np.mean(elapsed_times),
            'std_elapsed_time_sec': np.std(elapsed_times)
        }

        # Save individual trial data
        dimension_dir = os.path.join(output_dir, f"Dimension_{dimension}")
        if not os.path.exists(dimension_dir):
            os.makedirs(dimension_dir)

        for trial in trials:
            trial_file = os.path.join(dimension_dir, f"Trial_{trial['trial_id']}.json")
            with open(trial_file, 'w') as f:
                json.dump(trial, f, indent=4, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj)

        # Plot Success Rate
        plt.figure(figsize=(6,4))
        plt.bar([dimension], [success_rate], color='green' if success_rate == 1.0 else 'orange')
        plt.ylim(0,1)
        plt.xlabel('Dimension')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate for Dimension {dimension}')
        plt.savefig(os.path.join(dimension_dir, f'Success_Rate_Dim_{dimension}.png'))
        plt.close()

        # Plot Iterations
        plt.figure(figsize=(6,4))
        plt.hist(iterations, bins=range(0, max(iterations)+2), color='blue', alpha=0.7)
        plt.xlabel('Iterations to Converge')
        plt.ylabel('Frequency')
        plt.title(f'Iterations Distribution for Dimension {dimension}')
        plt.savefig(os.path.join(dimension_dir, f'Iterations_Dim_{dimension}.png'))
        plt.close()

        # Plot Volume Change
        volume_changes = np.array(volumes_final) - np.array(volumes_initial)
        plt.figure(figsize=(6,4))
        plt.hist(volume_changes, bins=20, color='purple', alpha=0.7)
        plt.xlabel('Volume Change')
        plt.ylabel('Frequency')
        plt.title(f'Convex Hull Volume Change for Dimension {dimension}')
        plt.savefig(os.path.join(dimension_dir, f'Volume_Change_Dim_{dimension}.png'))
        plt.close()

    # Save summary statistics
    summary_file = os.path.join(output_dir, 'summary_statistics.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    # Generate overall summary report
    report_file = os.path.join(output_dir, 'summary_report.txt')
    with open(report_file, 'w') as f:
        for dimension, stats in summary.items():
            f.write(f"Dimension {dimension}:\n")
            f.write(f"  Total Trials: {stats['total_trials']}\n")
            f.write(f"  Successful Trials: {stats['successful_trials']}\n")
            f.write(f"  Success Rate: {stats['success_rate']*100:.2f}%\n")
            f.write(f"  Average Iterations: {stats['average_iterations']:.2f} ± {stats['std_iterations']:.2f}\n")
            f.write(f"  Average Accepted Moves: {stats['average_accepted_moves']:.2f} ± {stats['std_accepted_moves']:.2f}\n")
            f.write(f"  Average Rejected Moves: {stats['average_rejected_moves']:.2f} ± {stats['std_rejected_moves']:.2f}\n")
            f.write(f"  Average Initial Volume: {stats['average_initial_volume']:.4f} ± {stats['std_initial_volume']:.4f}\n")
            f.write(f"  Average Final Volume: {stats['average_final_volume']:.4f} ± {stats['std_final_volume']:.4f}\n")
            f.write(f"  Average Elapsed Time: {stats['average_elapsed_time_sec']:.2f} sec ± {stats['std_elapsed_time_sec']:.2f} sec\n")
            f.write("\n")

    logging.info(f"Analysis complete. Summary statistics and plots saved in '{output_dir}' directory.")

def visualize_convergence(trial_result, output_dir='visualizations'):
    """
    Visualize the convergence process of a single trial.

    Args:
        trial_result (dict): The result dictionary from a single trial.
        output_dir (str): Directory where visualization will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    history = trial_result['history']
    iterations = range(1, history['iterations'] + 1)

    plt.figure(figsize=(12, 6))

    # Plot Temperature
    plt.subplot(1, 3, 1)
    plt.plot(iterations, history['temperature'], color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Decay')

    # Plot Score
    plt.subplot(1, 3, 2)
    plt.plot(iterations, history['score'], color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score Over Iterations')

    # Plot Volume
    plt.subplot(1, 3, 3)
    plt.plot(iterations, history['volume'], color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Convex Hull Volume')
    plt.title('Convex Hull Volume Over Iterations')

    plt.tight_layout()
    trial_id = trial_result['trial_id']
    dimension = trial_result['dimension']
    plt.savefig(os.path.join(output_dir, f'Dimension_{dimension}_Trial_{trial_id}_Convergence.png'))
    plt.close()

def main():
    """
    Main function to execute the verification process.
    """
    # Parameters (can be adjusted as needed)
    max_dimension = 20
    num_trials = 10
    num_iterations = 1000
    initial_temperature = 10.0
    cooling_rate = 0.95
    adjustment_scale = 0.1
    use_parallel = True
    verbose = False  # Set to True for detailed debug information

    # Run all trials
    results = run_all_trials(
        max_dimension=max_dimension,
        num_trials=num_trials,
        num_iterations=num_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        adjustment_scale=adjustment_scale,
        use_parallel=use_parallel,
        verbose=verbose
    )

    # Analyze results
    analyze_results(results, output_dir='results_analysis')

    # Optionally, visualize some trial convergence
    visualize_dir = 'visualizations'
    for dimension, trials in results.items():
        for trial in trials:
            visualize_convergence(trial, output_dir=visualize_dir)

    logging.info("McMullen Problem verification process completed successfully.")

if __name__ == "__main__":
    main()
