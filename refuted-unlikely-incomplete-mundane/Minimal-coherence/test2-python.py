import numpy as np
from scipy import stats

# Parameters
np.random.seed(42)
sample_size = 1000
trials = 50

# Path Space Function (Ensuring Symmetry)
def path_space(x, y):
    dist = np.linalg.norm(x - y)
    dim = len(x)
    perturbation = 0.01 * np.exp(-0.3 * dist) * (
        np.random.uniform(-1, 1, (dim, dim)) +
        np.random.uniform(-1, 1, (dim, dim)).T
    ) / (2 * (1 + 0.01 * dim))
    return np.eye(dim) + perturbation

# Property Computations with Error Bounds
def compute_property_with_error(dim, property_func, trials=50):
    results = np.array([property_func(dim) for _ in range(trials)])
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'ci_low': np.percentile(results, 2.5),
        'ci_high': np.percentile(results, 97.5)
    }

def test_reflexivity(dim):
    points = np.random.uniform(-1, 1, (sample_size, dim))
    paths = np.array([path_space(p, p) for p in points])
    deviations = np.array([np.linalg.norm(path - np.eye(dim)) for path in paths])
    return 1 - np.mean(deviations)

def test_symmetry(dim):
    points1 = np.random.uniform(-1, 1, (sample_size, dim))
    points2 = np.random.uniform(-1, 1, (sample_size, dim))
    paths1 = np.array([path_space(p1, p2) for p1, p2 in zip(points1, points2)])
    paths2 = np.array([path_space(p2, p1) for p1, p2 in zip(points1, points2)])
    deviations = np.array([np.linalg.norm(p1 - p2.T) for p1, p2 in zip(paths1, paths2)])
    return 1 - np.mean(deviations)

def test_transitivity(dim):
    points1 = np.random.uniform(-1, 1, (sample_size, dim))
    points2 = np.random.uniform(-1, 1, (sample_size, dim))
    points3 = np.random.uniform(-1, 1, (sample_size, dim))
    paths12 = np.array([path_space(p1, p2) for p1, p2 in zip(points1, points2)])
    paths23 = np.array([path_space(p2, p3) for p2, p3 in zip(points2, points3)])
    paths13 = np.array([path_space(p1, p3) for p1, p3 in zip(points1, points3)])
    deviations = np.array([np.linalg.norm(p12 @ p23 - p13) for p12, p23, p13 in zip(paths12, paths23, paths13)])
    return 1 - np.mean(deviations)

# Compute Decay Rates
def compute_decay_rates(values):
    return np.diff(np.log(values)) / np.diff(np.log(range(1, len(values) + 1)))

# Statistical Test for Phase Transitions
def test_phase_transition(values, potential_transition):
    before = values[:potential_transition]
    after = values[potential_transition:]
    return stats.ttest_ind(
        compute_decay_rates(before),
        compute_decay_rates(after)
    ).pvalue < 0.05

# Main Analysis
dimensions = range(1, 21)
results = {
    "Dimension": [],
    "Reflexivity": [],
    "Symmetry": [],
    "Transitivity": [],
    "DecayRates_R": [],
    "DecayRates_S": [],
    "DecayRates_T": [],
}

print("Starting Meta-Path Analysis with Error Bounds and Phase Detection...\n")

for dim in dimensions:
    print(f"Analyzing Dimension: {dim}")
    r_stats = compute_property_with_error(dim, test_reflexivity, trials)
    s_stats = compute_property_with_error(dim, test_symmetry, trials)
    t_stats = compute_property_with_error(dim, test_transitivity, trials)

    results["Dimension"].append(dim)
    results["Reflexivity"].append(r_stats['mean'])
    results["Symmetry"].append(s_stats['mean'])
    results["Transitivity"].append(t_stats['mean'])

    print(f"  Reflexivity: Mean = {r_stats['mean']:.5f}, CI = [{r_stats['ci_low']:.5f}, {r_stats['ci_high']:.5f}]")
    print(f"  Symmetry:    Mean = {s_stats['mean']:.5f}, CI = [{s_stats['ci_low']:.5f}, {s_stats['ci_high']:.5f}]")
    print(f"  Transitivity:Mean = {t_stats['mean']:.5f}, CI = [{t_stats['ci_low']:.5f}, {t_stats['ci_high']:.5f}]")

# Compute Decay Rates
results["DecayRates_R"] = compute_decay_rates(results["Reflexivity"])
results["DecayRates_S"] = compute_decay_rates(results["Symmetry"])
results["DecayRates_T"] = compute_decay_rates(results["Transitivity"])

# Test Phase Transitions
transition_R = test_phase_transition(results["Reflexivity"], 5)
transition_S = test_phase_transition(results["Symmetry"], 5)
transition_T = test_phase_transition(results["Transitivity"], 5)

print("\nPhase Transition Test Results:")
print(f"  Reflexivity Transition at d=5: {'Significant' if transition_R else 'Not Significant'}")
print(f"  Symmetry Transition at d=5:    {'Significant' if transition_S else 'Not Significant'}")
print(f"  Transitivity Transition at d=5:{'Significant' if transition_T else 'Not Significant'}")

print("\nAnalysis Complete.")
