from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict
from collections import defaultdict
import numpy as np

def get_solutions_for_z(z: int) -> List[Tuple[int, int]]:
    """Get all solutions for a z-value"""
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

def analyze_sqrt2_relationship(solutions: List[Tuple[int, int]]) -> Dict:
    """Analyze how close the y-range ratio is to √2"""
    if not solutions:
        return {}

    y_values = [y for _, y in solutions]
    actual_ratio = max(y_values) / min(y_values)
    sqrt2 = sqrt(2)

    # Analyze different powers of √2
    ratios = []
    for power in range(-4, 5):  # Check powers from -4 to 4
        target = sqrt2 ** power
        error = abs(actual_ratio - target)
        ratios.append((power, target, error))

    best_fit = min(ratios, key=lambda x: x[2])

    return {
        'actual_ratio': actual_ratio,
        'closest_sqrt2_power': best_fit[0],
        'closest_sqrt2_value': best_fit[1],
        'error': best_fit[2],
        'error_percentage': (best_fit[2] / best_fit[1]) * 100
    }

def analyze_family_sizes(max_z: int = 1_000_000, min_solutions: int = 20) -> Dict:
    """Analyze ratio patterns across different family sizes"""
    families = defaultdict(list)
    sqrt2_patterns = defaultdict(list)

    print(f"Analyzing families up to z={max_z:,}...")

    # Collect families
    for z in range(2, max_z + 1):
        solutions = get_solutions_for_z(z)
        if len(solutions) >= min_solutions:
            families[len(solutions)].append(z)
            sqrt2_analysis = analyze_sqrt2_relationship(solutions)
            sqrt2_patterns[len(solutions)].append(sqrt2_analysis)

            if z % 100000 == 0:
                print(f"Progress: {z/max_z*100:.1f}%")

    # Analyze patterns by family size
    results = {}
    for size, z_values in families.items():
        patterns = sqrt2_patterns[size]
        ratios = [p['actual_ratio'] for p in patterns]
        errors = [p['error'] for p in patterns]

        results[size] = {
            'count': len(z_values),
            'z_values': z_values,
            'ratio_stats': {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'min': min(ratios),
                'max': max(ratios)
            },
            'error_stats': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': min(errors),
                'max': max(errors)
            }
        }

    return results

def analyze_density_decay(families: Dict[int, Dict]) -> Dict:
    """Analyze how solution density decays with z"""
    decay_patterns = {}

    for size, data in families.items():
        z_values = data['z_values']
        densities = []

        for z in z_values:
            solutions = get_solutions_for_z(z)
            x_range = max(x for x,_ in solutions) - min(x for x,_ in solutions)
            density = size / x_range
            densities.append((z, density))

        # Fit decay curve
        z_array = np.array([z for z,_ in densities])
        d_array = np.array([d for _,d in densities])

        # Try different decay models
        # 1/z model
        fit_1_z = np.polyfit(1/z_array, d_array, 1)
        # 1/z² model
        fit_2_z = np.polyfit(1/(z_array*z_array), d_array, 1)
        # log model
        fit_log = np.polyfit(np.log(z_array), d_array, 1)

        # Calculate R² for each model
        r2_1_z = np.corrcoef(1/z_array, d_array)[0,1]**2
        r2_2_z = np.corrcoef(1/(z_array*z_array), d_array)[0,1]**2
        r2_log = np.corrcoef(np.log(z_array), d_array)[0,1]**2

        decay_patterns[size] = {
            'densities': densities,
            'models': {
                '1/z': {'coefficients': fit_1_z, 'r2': r2_1_z},
                '1/z²': {'coefficients': fit_2_z, 'r2': r2_2_z},
                'log': {'coefficients': fit_log, 'r2': r2_log}
            }
        }

    return decay_patterns

def main():
    # First analyze the √2 relationship in detail
    z_95_solutions = [330182, 617427, 652082, 700107, 780262,
                     819668, 899168, 920418, 946343]

    print("\n=== √2 RELATIONSHIP ANALYSIS FOR 95-SOLUTION FAMILIES ===")
    for z in z_95_solutions:
        solutions = get_solutions_for_z(z)
        analysis = analyze_sqrt2_relationship(solutions)
        print(f"\nZ = {z:,}")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"{key}: {value:.8f}")
            else:
                print(f"{key}: {value}")

    # Then analyze across all family sizes
    print("\n=== FAMILY SIZE ANALYSIS ===")
    family_analysis = analyze_family_sizes(1_000_000, 20)

    print("\nSize Distribution and Ratio Patterns:")
    for size in sorted(family_analysis.keys(), reverse=True):
        data = family_analysis[size]
        print(f"\nSize {size}:")
        print(f"Count: {data['count']:,}")
        print(f"Ratio mean: {data['ratio_stats']['mean']:.8f} ± {data['ratio_stats']['std']:.8f}")
        print(f"Error mean: {data['error_stats']['mean']:.8f} ± {data['error_stats']['std']:.8f}")

    # Finally analyze density decay
    print("\n=== DENSITY DECAY ANALYSIS ===")
    decay_analysis = analyze_density_decay(family_analysis)

    print("\nBest Fitting Decay Models:")
    for size, data in sorted(decay_analysis.items(), reverse=True):
        print(f"\nSize {size}:")
        models = data['models']
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        print(f"Best fit: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

if __name__ == "__main__":
    main()
