import numpy as np
from scipy import stats

def logical_consistency_test(trials=1000, dimensions=[4,5,6], sample_size=1000):
    """
    Tests logical consistency of phase transitions and ratio preservation.

    Key insight: If R changes sharply at d=5 AND S/R is constant,
    THEN S must change proportionally at d=5.
    """

    def path_space(x, y, dim):
        dist = np.linalg.norm(x - y)
        return np.eye(dim) + 0.01*np.exp(-0.3*dist)*np.random.uniform(-1, 1, (dim, dim))/(1 + 0.01*dim)

    results = {d: {'R': [], 'S': [], 'S/R': []} for d in dimensions}

    print("Running logical consistency test...")

    for d in dimensions:
        print(f"\nDimension {d}")
        for t in range(trials):
            if t % 100 == 0:
                print(f"Trial {t}")

            # Generate points
            points = np.random.uniform(-1, 1, (sample_size, d))

            # Measure R
            R_paths = [path_space(p, p, d) for p in points]
            R = 1 - np.mean([np.linalg.norm(p - np.eye(d)) for p in R_paths])

            # Measure S
            points2 = np.random.uniform(-1, 1, (sample_size, d))
            S_paths1 = [path_space(p1, p2, d) for p1, p2 in zip(points, points2)]
            S_paths2 = [path_space(p2, p1, d) for p1, p2 in zip(points, points2)]
            S = 1 - np.mean([np.linalg.norm(p1 - p2.T) for p1, p2 in zip(S_paths1, S_paths2)])

            # Store results
            results[d]['R'].append(R)
            results[d]['S'].append(S)
            results[d]['S/R'].append(S/R)

    # Analysis
    print("\nAnalyzing results...")

    # 1. Test for R phase transition
    R_change = np.mean(results[5]['R']) - np.mean(results[4]['R'])
    R_next_change = np.mean(results[6]['R']) - np.mean(results[5]['R'])
    R_transition_significance = abs(R_change) > 2*abs(R_next_change)

    # 2. Test for S/R ratio consistency
    ratios = [results[d]['S/R'] for d in dimensions]
    ratio_variances = [np.var(r) for r in ratios]
    ratio_means = [np.mean(r) for r in ratios]
    ratio_consistent = max(ratio_means) - min(ratio_means) < 0.001

    # 3. Test for proportional S transition
    S_change = np.mean(results[5]['S']) - np.mean(results[4]['S'])
    S_next_change = np.mean(results[6]['S']) - np.mean(results[5]['S'])
    S_transition_proportional = abs(S_change/R_change - np.mean(results[5]['S/R'])) < 0.001

    # Detailed results
    print("\nDetailed Results:")
    print("\nR values:")
    for d in dimensions:
        print(f"d={d}: {np.mean(results[d]['R']):.6f} ± {np.std(results[d]['R']):.6f}")

    print("\nS values:")
    for d in dimensions:
        print(f"d={d}: {np.mean(results[d]['S']):.6f} ± {np.std(results[d]['S']):.6f}")

    print("\nS/R ratios:")
    for d in dimensions:
        print(f"d={d}: {np.mean(results[d]['S/R']):.6f} ± {np.std(results[d]['S/R']):.6f}")

    print("\nTransition Analysis:")
    print(f"R change at d=5: {R_change:.6f}")
    print(f"R change at d=6: {R_next_change:.6f}")
    print(f"S change at d=5: {S_change:.6f}")
    print(f"Ratio variation: {max(ratio_means) - min(ratio_means):.6f}")

    # Logical conclusion
    mathematica_consistent = (R_transition_significance and
                            ratio_consistent and
                            S_transition_proportional)

    python_consistent = (not R_transition_significance or
                        not ratio_consistent or
                        not S_transition_proportional)

    print("\nLogical Consistency Results:")
    print(f"R transition detected: {R_transition_significance}")
    print(f"Ratio consistency maintained: {ratio_consistent}")
    print(f"S transition proportional: {S_transition_proportional}")
    print(f"Mathematica model consistency: {mathematica_consistent}")
    print(f"Python model consistency: {python_consistent}")

    if mathematica_consistent and not python_consistent:
        print("\nCONCLUSION: Mathematica results are more likely correct.")
    elif python_consistent and not mathematica_consistent:
        print("\nCONCLUSION: Python results are more likely correct.")
    else:
        print("\nCONCLUSION: Results inconclusive, need additional tests.")

    return {
        'R_transition': R_transition_significance,
        'ratio_consistency': ratio_consistent,
        'S_proportionality': S_transition_proportional,
        'raw_results': results,
        'mathematica_consistent': mathematica_consistent,
        'python_consistent': python_consistent
    }

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    results = logical_consistency_test()
