import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
import warnings

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

# Property Computations
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

# Phase Transition Detection
def detect_phase_transitions(values, dimensions):
    log_values = np.log(values) / np.log(dimensions)
    n = len(log_values)
    log_p = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 2, n):
            segment = log_values[i:j]
            mu = np.mean(segment)
            sigma = np.std(segment) + 1e-6
            log_p[i, j] = norm.logpdf(segment, loc=mu, scale=sigma).sum()

    return find_optimal_segments(log_p, max_segments=3)

# Error Tracking and Stability
def track_accumulated_error(results, r, s, t):
    if len(results["Dimension"]) > 1:
        last = len(results["Dimension"]) - 1
        accumulated_error = np.array([
            np.abs(r - results["Reflexivity"][last]),
            np.abs(s - results["Symmetry"][last]),
            np.abs(t - results["Transitivity"][last])
        ])
        if np.any(accumulated_error > 1e-3):
            warnings.warn(f"Accumulated error exceeds threshold at dimension {results['Dimension'][-1]}")
        return accumulated_error
    return np.zeros(3)

# Main Analysis Loop
dimensions = range(1, 21)
results = {
    "Dimension": [],
    "Reflexivity": [],
    "Symmetry": [],
    "Transitivity": []
}

print("Starting Meta-Path Analysis...")

for dim in dimensions:
    print(f"\nAnalyzing Dimension: {dim}")
    r = test_reflexivity(dim)
    s = test_symmetry(dim)
    t = test_transitivity(dim)

    results["Dimension"].append(dim)
    results["Reflexivity"].append(r)
    results["Symmetry"].append(s)
    results["Transitivity"].append(t)

    print(f"  Reflexivity: {r:.5f}")
    print(f"  Symmetry:    {s:.5f}")
    print(f"  Transitivity:{t:.5f}")

    # Track numerical errors
    track_accumulated_error(results, r, s, t)

# Calculate Ratios
results["S/R"] = np.array(results["Symmetry"]) / np.array(results["Reflexivity"])
results["T/R"] = np.array(results["Transitivity"]) / np.array(results["Reflexivity"])
results["S/T"] = np.array(results["Symmetry"]) / np.array(results["Transitivity"])

print("\nAnalysis Complete.")
print("\nResults:")
for dim, r, s, t in zip(results["Dimension"], results["Reflexivity"], results["Symmetry"], results["Transitivity"]):
    print(f"Dimension {dim}: Reflexivity = {r:.5f}, Symmetry = {s:.5f}, Transitivity = {t:.5f}")

print("\nRatios (Mean across Dimensions):")
print(f"  S/R: {np.mean(results['S/R']):.5f}")
print(f"  T/R: {np.mean(results['T/R']):.5f}")
print(f"  S/T: {np.mean(results['S/T']):.5f}")
