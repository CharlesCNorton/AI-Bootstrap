import numpy as np
from scipy import stats
import time
from multiprocessing import Pool
import pandas as pd

def path_space(x, y, precision):
    x = x.astype(precision)
    y = y.astype(precision)
    dist = np.linalg.norm(x - y)
    dim = len(x)
    base = np.eye(dim, dtype=precision)
    perturbation = (0.01 * np.exp(-0.3 * dist) *
                   np.random.uniform(-1, 1, (dim, dim)).astype(precision))
    return base + perturbation / (1 + 0.01 * dim)

def compute_properties(dim, sample_size, precision):
    points = np.random.uniform(-1, 1, (sample_size, dim)).astype(precision)

    # Reflexivity
    ref_paths = np.array([path_space(p, p, precision) for p in points])
    reflexivity = 1 - np.mean([np.linalg.norm(p - np.eye(dim)) for p in ref_paths])

    # Symmetry
    points2 = np.random.uniform(-1, 1, (sample_size, dim)).astype(precision)
    sym_paths1 = np.array([path_space(p1, p2, precision)
                          for p1, p2 in zip(points, points2)])
    sym_paths2 = np.array([path_space(p2, p1, precision)
                          for p1, p2 in zip(points, points2)])
    symmetry = 1 - np.mean([np.linalg.norm(p1 - p2.T)
                           for p1, p2 in zip(sym_paths1, sym_paths2)])

    # Transitivity
    points3 = np.random.uniform(-1, 1, (sample_size, dim)).astype(precision)
    trans_paths12 = np.array([path_space(p1, p2, precision)
                             for p1, p2 in zip(points, points2)])
    trans_paths23 = np.array([path_space(p2, p3, precision)
                             for p2, p3 in zip(points2, points3)])
    trans_paths13 = np.array([path_space(p1, p3, precision)
                             for p1, p3 in zip(points, points3)])
    transitivity = 1 - np.mean([np.linalg.norm(p12 @ p23 - p13)
                              for p12, p23, p13 in zip(trans_paths12,
                                                     trans_paths23,
                                                     trans_paths13)])

    return reflexivity, symmetry, transitivity

def run_single_configuration(config):
    sample_size, trial_count, precision, dim = config
    start_time = time.time()

    # Run trials
    trial_results = np.array([compute_properties(dim, sample_size, precision)
                            for _ in range(trial_count)])

    # Compute statistics
    means = np.mean(trial_results, axis=0)
    stds = np.std(trial_results, axis=0)

    # Check for phase transitions
    if dim > 1:
        prev_results = np.array([compute_properties(dim-1, sample_size, precision)
                               for _ in range(trial_count)])
        prev_means = np.mean(prev_results, axis=0)
        phase_transition = any(stats.ttest_ind(trial_results, prev_results).pvalue < 0.001)
    else:
        phase_transition = False

    # Numerical stability metric
    stability = np.mean(np.abs(np.diff(trial_results, axis=0)))

    computation_time = time.time() - start_time

    return {
        'sample_size': sample_size,
        'trial_count': trial_count,
        'precision': precision.__name__,
        'dimension': dim,
        'computation_time': computation_time,
        'reflexivity_mean': means[0],
        'reflexivity_std': stds[0],
        'symmetry_mean': means[1],
        'symmetry_std': stds[1],
        'transitivity_mean': means[2],
        'transitivity_std': stds[2],
        'phase_transition_detected': phase_transition,
        'numerical_stability_metric': stability
    }

def analyze_convergence(group):
    return pd.Series({
        'value_range': group.max() - group.min(),
        'std_range': group.std(),
        'convergence_metric': (group.max() - group.min()) / group.mean()
    })

def run_comparative_analysis(dimensions=[1,2,3,4,5,10,15,20],
                           sample_sizes=[1000, 5000, 10000, 50000],
                           trial_counts=[50, 100, 500, 1000],
                           precision_levels=[np.float32, np.float64]):

    # Generate all configurations
    configurations = [(s, t, p, d)
                     for s in sample_sizes
                     for t in trial_counts
                     for p in precision_levels
                     for d in dimensions]

    print(f"Starting analysis with {len(configurations)} total configurations...")

    # Run parallel computation
    with Pool() as pool:
        all_results = pool.map(run_single_configuration, configurations)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Analyze convergence
    convergence_analysis = df.groupby('dimension').agg({
        'reflexivity_mean': analyze_convergence,
        'symmetry_mean': analyze_convergence,
        'transitivity_mean': analyze_convergence
    })

    print("\nAnalysis Complete. Saving results...")

    # Save detailed results
    df.to_csv('meta_path_comparative_analysis.csv')
    convergence_analysis.to_csv('meta_path_convergence_analysis.csv')

    return df, convergence_analysis

if __name__ == "__main__":
    print("Starting Comprehensive Meta-Path Analysis...")
    results_df, convergence_df = run_comparative_analysis()

    print("\nKey Findings:")
    print("1. Computation Time vs Precision:")
    print(results_df.groupby('precision')['computation_time'].mean())

    print("\n2. Phase Transition Detection Rate:")
    print(results_df.groupby('dimension')['phase_transition_detected'].mean())

    print("\n3. Numerical Stability by Precision Level:")
    print(results_df.groupby('precision')['numerical_stability_metric'].mean())

    print("\n4. Convergence Analysis Summary:")
    print(convergence_df.mean())
