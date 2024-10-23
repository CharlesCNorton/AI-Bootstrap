import os
import numpy as np
from scipy.spatial import ConvexHull
import random
import logging
import time
import math

# Limit OpenBLAS to use only one thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Configure Logging
logging.basicConfig(
    filename='mcmullen_verification.log',
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
    Verify if all points are vertices of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception as e:
        logging.debug(f"ConvexHull computation failed: {e}")
        return False


def calculate_energy(points):
    """
    Calculate the energy as the sum of distances of points to the convex hull.
    """
    try:
        hull = ConvexHull(points)
        hull_equations = hull.equations
        distances = np.abs(np.dot(points, hull_equations[:, :-1].T) + hull_equations[:, -1])
        min_distances = np.min(distances, axis=1)
        energy = np.sum(min_distances)
        return energy
    except Exception as e:
        logging.debug(f"Energy calculation failed: {e}")
        return float('inf')


def generate_non_convex_points(num_points, dimension):
    """
    Generate points not in convex position by placing some inside the convex hull.
    """
    # Number of interior points
    num_interior = max(1, num_points // 4)
    num_hull_points = num_points - num_interior

    # Generate hull points
    hull_points = np.random.uniform(low=0, high=1, size=(num_hull_points, dimension))

    # Generate interior points as the mean of hull points plus some noise
    center_point = np.mean(hull_points, axis=0)
    interior_points = center_point + np.random.normal(scale=0.05, size=(num_interior, dimension))

    # Combine hull and interior points
    points = np.vstack([hull_points, interior_points])
    return points


def simulated_annealing(
    points,
    num_iterations=1000,
    initial_temperature=10.0,
    cooling_rate=0.99,
    adjustment_scale=0.1,
    verbose=False
):
    """
    Perform simulated annealing to move points into convex position.
    """
    current_points = np.copy(points)
    current_energy = calculate_energy(current_points)
    initial_energy = current_energy

    temperature = initial_temperature
    history = {
        'temperature': [],
        'energy': [],
        'accepted_moves': 0,
        'rejected_moves': 0,
        'iterations': 0
    }

    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        # Record history
        history['temperature'].append(temperature)
        history['energy'].append(current_energy)

        # Make a small random adjustment to a random point
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=adjustment_scale, size=current_points.shape[1])
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment

        # Compute energy of new configuration
        new_energy = calculate_energy(new_points)

        # Compute energy difference
        delta_energy = new_energy - current_energy

        # Compute acceptance probability
        if delta_energy < 0:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp(-delta_energy / temperature) if temperature > 0 else 0.0

        # Decide whether to accept the new configuration
        if random.random() < acceptance_probability:
            current_points = new_points
            current_energy = new_energy
            history['accepted_moves'] += 1
            if verbose:
                logging.debug(f"Iteration {iteration}: Accepted move. New energy: {current_energy}")
        else:
            history['rejected_moves'] += 1
            if verbose:
                logging.debug(f"Iteration {iteration}: Rejected move. Energy remains: {current_energy}")

        # Cool down the temperature
        temperature *= cooling_rate
        history['iterations'] = iteration

        # If energy is close to zero, break early
        if current_energy < 1e-6:
            if verbose:
                logging.debug(f"Iteration {iteration}: Convex configuration achieved.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    result = {
        'optimized_points': current_points,
        'final_energy': current_energy,
        'initial_energy': initial_energy,
        'history': history,
        'elapsed_time_sec': elapsed_time
    }

    return result


def run_trial(
    dimension,
    trial_id,
    num_iterations=1000,
    initial_temperature=10.0,
    cooling_rate=0.99,
    adjustment_scale=0.1,
    verbose=False
):
    """
    Run a single trial of simulated annealing for a given dimension.
    """
    num_points = 2 * dimension + 1
    # Generate initial points that are not in convex position
    points = generate_non_convex_points(num_points, dimension)

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
    cooling_rate=0.99,
    adjustment_scale=0.1,
    use_parallel=True,
    verbose=False
):
    """
    Run multiple trials across multiple dimensions to verify the McMullen Problem conjecture.
    """
    results = {}
    start_time = time.time()

    logging.info("Starting all trials for McMullen Problem verification.")

    dimensions = range(3, max_dimension + 1)

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(4, os.cpu_count())  # Limit the number of worker threads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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


def analyze_results(results):
    """
    Analyze the results of the trials and print summary statistics to the console.
    """
    summary = {}

    for dimension, trials in results.items():
        total_trials = len(trials)
        successful_trials = sum(1 for trial in trials if trial['is_convex'])
        success_rate = successful_trials / total_trials
        iterations = [trial['history']['iterations'] for trial in trials]
        accepted_moves = [trial['history']['accepted_moves'] for trial in trials]
        rejected_moves = [trial['history']['rejected_moves'] for trial in trials]
        initial_energies = [trial['initial_energy'] for trial in trials]
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
            'average_initial_energy': np.mean(initial_energies),
            'std_initial_energy': np.std(initial_energies),
            'average_elapsed_time_sec': np.mean(elapsed_times),
            'std_elapsed_time_sec': np.std(elapsed_times)
        }

    # Print summary statistics
    print("\nSummary of Results:")
    for dimension, stats in summary.items():
        print(f"Dimension {dimension}:")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Successful Trials: {stats['successful_trials']}")
        print(f"  Success Rate: {stats['success_rate'] * 100:.2f}%")
        print(f"  Average Iterations: {stats['average_iterations']:.2f} ± {stats['std_iterations']:.2f}")
        print(f"  Average Accepted Moves: {stats['average_accepted_moves']:.2f} ± {stats['std_accepted_moves']:.2f}")
        print(f"  Average Rejected Moves: {stats['average_rejected_moves']:.2f} ± {stats['std_rejected_moves']:.2f}")
        print(f"  Average Initial Energy: {stats['average_initial_energy']:.4f} ± {stats['std_initial_energy']:.4f}")
        print(f"  Average Elapsed Time: {stats['average_elapsed_time_sec']:.2f} sec ± {stats['std_elapsed_time_sec']:.2f}")
        print("\n")


# Example usage:
if __name__ == "__main__":
    # Run all trials
    results = run_all_trials(
        max_dimension=20,
        num_trials=10,
        num_iterations=1000,
        initial_temperature=10.0,
        cooling_rate=0.99,
        adjustment_scale=0.1,
        use_parallel=True,
        verbose=False
    )

    # Analyze results and print them
    analyze_results(results)
