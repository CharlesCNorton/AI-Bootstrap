import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict
import math
from itertools import combinations

def C(n):
    """Calculate minimal coherence conditions for n-categories"""
    if n < 2:
        raise ValueError("n must be >= 2")

    if n <= 3:
        return n - 1  # Foundational phase
    elif n <= 5:
        return 2*n - 3  # Transitional phase
    else:
        return 2*n - 1  # Linear phase

def analyze_higher_order_patterns(max_n=50):
    """Analyze higher-order differences and patterns"""
    values = [C(n) for n in range(2, max_n)]

    # Calculate differences up to 4th order
    diffs = {
        '1st': np.diff(values),
        '2nd': np.diff(np.diff(values)),
        '3rd': np.diff(np.diff(np.diff(values))),
        '4th': np.diff(np.diff(np.diff(np.diff(values))))
    }

    # Analyze each order
    pattern_analysis = {}
    for order, diff in diffs.items():
        pattern_analysis[order] = {
            'mean': np.mean(diff),
            'std': np.std(diff),
            'zeros': np.sum(diff == 0),
            'distinct_values': len(set(diff))
        }

    return pattern_analysis

def analyze_transition_points():
    """Detailed analysis of transition points"""
    n_range = range(2, 15)
    values = [C(n) for n in n_range]
    diffs = np.diff(values)

    # Detect transitions
    transitions = []
    for i in range(1, len(diffs)):
        if abs(diffs[i] - diffs[i-1]) > 0.1:
            transitions.append(i + 2)  # +2 due to indexing and diff

    # Analyze properties around transitions
    transition_analysis = {}
    for t in transitions:
        # Ensure we don't go below n=2
        before_start = max(2, t-2)
        before = [C(n) for n in range(before_start, t)]
        after = [C(n) for n in range(t, t+2)]

        # Calculate growth rates safely
        before_growth = np.mean(np.diff(before)) if len(before) > 1 else 0
        after_growth = np.mean(np.diff(after)) if len(after) > 1 else 0
        growth_ratio = after_growth / before_growth if before_growth != 0 else float('inf')

        transition_analysis[t] = {
            'before_growth': before_growth,
            'after_growth': after_growth,
            'growth_ratio': growth_ratio,
            'value_jump': C(t) - C(t-1) if t > 2 else C(t) - 0
        }

    return transition_analysis

def analyze_symmetries():
    """Investigate potential symmetries in the sequence"""
    values = [C(n) for n in range(2, 20)]

    # Test for various symmetry properties
    symmetry_tests = {
        'reflection': [],
        'translation': [],
        'scaling': []
    }

    # Test reflection symmetry around midpoints
    for i in range(2, len(values)-2):
        left = values[i-2:i]
        right = values[i+1:i+3]
        symmetry_tests['reflection'].append(
            np.allclose(np.diff(left), -np.diff(right[::-1]))
        )

    # Test translation symmetry
    for shift in range(1, 4):
        diffs = np.array(values[shift:]) - np.array(values[:-shift])
        symmetry_tests['translation'].append(
            np.std(diffs) < 0.1
        )

    # Test scaling symmetry
    for scale in [2, 3, 4]:
        ratios = np.array(values[scale:]) / np.array(values[:-scale])
        symmetry_tests['scaling'].append(
            np.std(ratios) < 0.1
        )

    return symmetry_tests

def number_theoretic_analysis():
    """Analyze number theoretic properties"""
    values = [C(n) for n in range(2, 20)]

    properties = {
        'parity': [],
        'divisibility': defaultdict(list),
        'congruences': defaultdict(list)
    }

    # Analyze parity pattern
    properties['parity'] = [v % 2 for v in values]

    # Test divisibility by small primes
    for prime in [2, 3, 5, 7]:
        properties['divisibility'][prime] = [v % prime == 0 for v in values]

    # Test congruences modulo small numbers
    for mod in [2, 3, 4, 5]:
        properties['congruences'][mod] = [v % mod for v in values]

    return properties

def comprehensive_analysis():
    """Complete analysis suite"""
    from scipy import stats  # Move import here to fix scope

    results = {}

    # Basic analysis
    n_values = range(2, 50)
    c_values = [C(n) for n in n_values]

    # 1. Core Analysis
    diffs = np.diff(c_values)
    growth_rate = np.mean(diffs[-10:])  # Stable phase growth rate

    # 2. Statistical Properties
    stable_x = range(len(c_values[8:]))
    stable_y = c_values[8:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(stable_x, stable_y)

    # 3. Higher Order Patterns
    higher_patterns = analyze_higher_order_patterns()

    # 4. Transition Analysis
    transitions = analyze_transition_points()

    # 5. Symmetry Analysis
    symmetries = analyze_symmetries()

    # 6. Number Theory
    number_theory = number_theoretic_analysis()

    # Print Results
    print("\nComprehensive Analysis Results:")

    print("\n1. Core Properties:")
    print(f"Growth rate (stable phase): {growth_rate:.3f}")
    print(f"Linear fit R²: {r_value**2:.6f}")

    print("\n2. Higher Order Patterns:")
    for order, stats in higher_patterns.items():
        print(f"\n{order} order differences:")
        for key, value in stats.items():
            print(f"{key}: {value:.3f}")

    print("\n3. Transition Points:")
    for point, analysis in transitions.items():
        print(f"\nTransition at n={point}:")
        for key, value in analysis.items():
            print(f"{key}: {value:.3f}")

    print("\n4. Symmetry Properties:")
    for sym_type, tests in symmetries.items():
        print(f"\n{sym_type} symmetry:")
        print(f"Tests passed: {sum(tests)}/{len(tests)}")

    print("\n5. Number Theoretic Properties:")
    print("Parity pattern:", number_theory['parity'][:10], "...")
    for mod, values in number_theory['congruences'].items():
        print(f"Modulo {mod}:", values[:10], "...")

    return {
        'core': {
            'growth_rate': growth_rate,
            'r_squared': r_value**2
        },
        'higher_patterns': higher_patterns,
        'transitions': transitions,
        'symmetries': symmetries,
        'number_theory': number_theory
    }

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = comprehensive_analysis()
