from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict
from collections import defaultdict
import numpy as np

def get_95_solution_family(z: int) -> List[Tuple[int, int]]:
    """Get all solutions for a specific z-value known to have 95 solutions"""
    solutions = []
    z_squared_plus_1 = z*z + 1
    max_x = isqrt(z_squared_plus_1 - 1)

    for x in range(2, max_x + 1):
        y_squared = z_squared_plus_1 - x*x
        if y_squared > 0:
            y = isqrt(y_squared)
            if y*y == y_squared and y > x:
                solutions.append((x, y))

    return sorted(solutions)

def analyze_95_solution_structure(z_values: List[int]):
    """Analyze the structure of known 95-solution families"""
    print(f"\nAnalyzing structure of 95-solution families...")

    # Known z-values with 95 solutions
    results = {}
    for z in z_values:
        solutions = get_95_solution_family(z)

        # Calculate key metrics
        x_values = [x for x,_ in solutions]
        y_values = [y for _,y in solutions]
        x_diffs = np.diff(x_values)
        y_diffs = np.diff(y_values)

        # Analyze solution spacing
        results[z] = {
            'x_min_gap': int(min(x_diffs)),
            'x_max_gap': int(max(x_diffs)),
            'x_median_gap': int(np.median(x_diffs)),
            'y_min_gap': int(min(y_diffs)),
            'y_max_gap': int(max(y_diffs)),
            'y_median_gap': int(np.median(y_diffs)),
            'x_range_ratio': max(x_values) / min(x_values),
            'y_range_ratio': max(y_values) / min(y_values),
            'solution_density': 95 / (max(x_values) - min(x_values))
        }

    return results

def main():
    # The nine z-values we found with 95 solutions
    z_values = [330182, 617427, 652082, 700107, 780262,
                819668, 899168, 920418, 946343]

    analysis = analyze_95_solution_structure(z_values)

    print("\n=== 95-SOLUTION FAMILY ANALYSIS ===")

    # Print individual family analyses
    for z, metrics in analysis.items():
        print(f"\nZ = {z:,}")
        for key, value in metrics.items():
            if 'ratio' in key or 'density' in key:
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value:,}")

    # Look for common patterns across all 95-solution families
    print("\n=== COMMON PATTERNS ===")
    metrics = list(analysis.values())

    for key in metrics[0].keys():
        values = [m[key] for m in metrics]
        print(f"\n{key}:")
        print(f"Min: {min(values):,.6f}")
        print(f"Max: {max(values):,.6f}")
        print(f"Mean: {np.mean(values):,.6f}")
        print(f"Std: {np.std(values):,.6f}")

if __name__ == "__main__":
    main()
