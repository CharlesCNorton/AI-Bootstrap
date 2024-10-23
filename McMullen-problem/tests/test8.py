import numpy as np
from scipy.spatial import ConvexHull
import random
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from math import comb

# Configure Logging
logging.basicConfig(
    filename='mcmullen_verification.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def verify_convex_position(points):
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception as e:
        logging.debug(f"ConvexHull computation failed: {e}")
        return False

def is_simplicial_polytope(hull):
    d = hull.points.shape[1]
    for simplex in hull.simplices:
        if len(simplex) != d:
            return False
    return True

def compute_f_vector(hull):
    from itertools import combinations
    d = hull.points.shape[1]
    simplices = hull.simplices
    faces = {}
    for i in range(d):
        faces[i] = set()
    for simplex in simplices:
        for i in range(1, d):
            for face in combinations(simplex, i+1):
                faces[i].add(tuple(sorted(face)))
    f_vector = [0] * (d + 1)
    f_vector[0] = len(hull.vertices)
    for i in range(1, d):
        f_vector[i] = len(faces[i])
    f_vector[d] = 1
    return f_vector

def compute_h_vector(f_vector, d):
    f = [1] + f_vector
    h_vector = []
    for i in range(0, d + 1):
        h_i = 0
        for j in range(0, i + 1):
            sign = (-1) ** (i - j)
            coeff = comb(d - j, d - i)
            h_i += sign * coeff * f[j]
        h_vector.append(h_i)
    return h_vector

def compute_g_vector(h_vector):
    g_vector = [h_vector[0]]
    for i in range(1, len(h_vector) // 2 + 1):
        g_value = h_vector[i] - h_vector[i - 1]
        g_vector.append(g_value)
    return g_vector

def verify_g_theorem(g_vector):
    return all(g >= 0 for g in g_vector)

def calculate_energy(points):
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
    num_interior = num_points // 2
    num_hull_points = num_points - num_interior
    hull_points = np.random.uniform(low=0, high=1, size=(num_hull_points, dimension))
    center_point = np.mean(hull_points, axis=0)
    interior_points = center_point + np.random.normal(scale=0.01, size=(num_interior, dimension))
    points = np.vstack([hull_points, interior_points])
    return points

def simulated_annealing(
    points,
    num_iterations=1000,
    initial_temperature=20.0,
    cooling_rate=0.995,
    adjustment_scale=0.01,
    verbose=False
):
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
        history['temperature'].append(temperature)
        history['energy'].append(current_energy)
        point_idx = random.randint(0, len(current_points) - 1)
        adjustment = np.random.normal(scale=adjustment_scale, size=current_points.shape[1])
        new_points = np.copy(current_points)
        new_points[point_idx] += adjustment
        new_energy = calculate_energy(new_points)
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp(-delta_energy / temperature) if temperature > 0 else 0.0
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
        temperature *= cooling_rate
        history['iterations'] = iteration
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
    initial_temperature=20.0,
    cooling_rate=0.995,
    adjustment_scale=0.01,
    verbose=False
):
    num_points = 2 * dimension + 1
    points = generate_non_convex_points(num_points, dimension)
    logging.info(f"Starting Trial {trial_id} for Dimension {dimension}")
    result = simulated_annealing(
        points,
        num_iterations=num_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        adjustment_scale=adjustment_scale,
        verbose=verbose
    )
    is_convex = verify_convex_position(result['optimized_points'])
    result['is_convex'] = is_convex
    result['trial_id'] = trial_id
    result['dimension'] = dimension
    if is_convex:
        hull = ConvexHull(result['optimized_points'])
        is_simplicial = is_simplicial_polytope(hull)
        result['is_simplicial'] = is_simplicial
        if is_simplicial:
            f_vector = compute_f_vector(hull)
            h_vector = compute_h_vector(f_vector, dimension)
            g_vector = compute_g_vector(h_vector)
            satisfies_g_theorem = verify_g_theorem(g_vector)
            result['f_vector'] = f_vector
            result['h_vector'] = h_vector
            result['g_vector'] = g_vector
            result['satisfies_g_theorem'] = satisfies_g_theorem
            if satisfies_g_theorem:
                logging.info(f"Trial {trial_id} for Dimension {dimension} succeeded and satisfies the g-theorem.")
            else:
                logging.warning(f"Trial {trial_id} for Dimension {dimension} failed to satisfy the g-theorem.")
        else:
            result['satisfies_g_theorem'] = False
            logging.warning(f"Trial {trial_id} for Dimension {dimension} does not produce a simplicial polytope.")
    else:
        result['is_simplicial'] = False
        result['satisfies_g_theorem'] = False
        logging.warning(f"Trial {trial_id} for Dimension {dimension} failed to achieve convex configuration.")
    return result

def run_all_trials(
    max_dimension=20,
    num_trials=10,
    num_iterations=1000,
    initial_temperature=20.0,
    cooling_rate=0.995,
    adjustment_scale=0.01,
    use_parallel=True,
    verbose=False
):
    results = {}
    start_time = time.time()
    logging.info("Starting all trials for McMullen Conjecture verification.")
    dimensions = range(3, max_dimension + 1)
    if use_parallel:
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
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
    summary = {}
    for dimension, trials in results.items():
        total_trials = len(trials)
        successful_trials = sum(1 for trial in trials if trial['satisfies_g_theorem'])
        convex_trials = sum(1 for trial in trials if trial['is_convex'])
        simplicial_trials = sum(1 for trial in trials if trial.get('is_simplicial', False))
        success_rate = successful_trials / total_trials
        iterations = [trial['history']['iterations'] for trial in trials]
        accepted_moves = [trial['history']['accepted_moves'] for trial in trials]
        rejected_moves = [trial['history']['rejected_moves'] for trial in trials]
        initial_energies = [trial['initial_energy'] for trial in trials]
        elapsed_times = [trial['elapsed_time_sec'] for trial in trials]
        summary[dimension] = {
            'total_trials': total_trials,
            'convex_trials': convex_trials,
            'simplicial_trials': simplicial_trials,
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
    print("\nSummary of Results:")
    for dimension, stats in summary.items():
        print(f"Dimension {dimension}:")
        print(f"  Total Trials: {stats['total_trials']}")
        print(f"  Convex Trials: {stats['convex_trials']}")
        print(f"  Simplicial Trials: {stats['simplicial_trials']}")
        print(f"  Successful Trials (g-theorem satisfied): {stats['successful_trials']}")
        print(f"  Success Rate: {stats['success_rate'] * 100:.2f}%")
        print(f"  Average Iterations: {stats['average_iterations']:.2f} ± {stats['std_iterations']:.2f}")
        print(f"  Average Accepted Moves: {stats['average_accepted_moves']:.2f} ± {stats['std_accepted_moves']:.2f}")
        print(f"  Average Rejected Moves: {stats['average_rejected_moves']:.2f} ± {stats['std_rejected_moves']:.2f}")
        print(f"  Average Initial Energy: {stats['average_initial_energy']:.4f} ± {stats['std_initial_energy']:.4f}")
        print(f"  Average Elapsed Time: {stats['average_elapsed_time_sec']:.2f} sec ± {stats['std_elapsed_time_sec']:.2f}")
        print("\n")

if __name__ == "__main__":
    results = run_all_trials(
        max_dimension=20,
        num_trials=10,
        num_iterations=2000,
        initial_temperature=20.0,
        cooling_rate=0.995,
        adjustment_scale=0.05,
        use_parallel=True,
        verbose=False
    )
    analyze_results(results)
