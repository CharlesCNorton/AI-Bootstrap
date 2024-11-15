from math import isqrt, sqrt
from collections import defaultdict
from typing import List, Tuple
import time

def find_solutions(z: int) -> List[Tuple[int, int]]:
    """Find all (x, y) pairs satisfying x² + y² = z² + 1 for a given z"""
    solutions = []
    for x in range(1, z):
        y_squared = z*z + 1 - x*x
        if y_squared <= 0:
            break
        y = isqrt(y_squared)
        if y*y == y_squared and y > x:
            solutions.append((x, y))
    return solutions

def analyze_z(z: int) -> dict:
    """Analyze solutions for a given z value"""
    solutions = find_solutions(z)
    if not solutions:
        return None

    y_values = [y for _, y in solutions]
    if y_values:
        ratio = max(y_values) / min(y_values)
        error_from_sqrt2 = abs(ratio - sqrt(2))

        return {
            'z': z,
            'solutions': len(solutions),
            'y_max/y_min': ratio,
            'error_from_sqrt2': error_from_sqrt2
        }
    return None

def main():
    print("Starting analysis...")
    start_time = time.time()

    # Track family sizes
    family_sizes = defaultdict(int)

    # Track the first occurrence of each family size
    first_occurrence = {}

    # Track statistics for each family size
    family_stats = defaultdict(lambda: {'count': 0, 'ratio_sum': 0, 'error_sum': 0})

    # Progress tracking
    milestone = 100000
    next_milestone = milestone

    max_z = 2000000

    for z in range(2, max_z + 1):
        result = analyze_z(z)

        if result:
            size = result['solutions']
            family_sizes[size] += 1

            # Update statistics
            family_stats[size]['count'] += 1
            family_stats[size]['ratio_sum'] += result['y_max/y_min']
            family_stats[size]['error_sum'] += result['error_from_sqrt2']

            if size not in first_occurrence:
                first_occurrence[size] = z
                print(f"\nFirst occurrence of family size {size} at z={z}")
                print(f"y_max/y_min ratio: {result['y_max/y_min']:.8f}")
                print(f"Error from √2: {result['error_from_sqrt2']:.8f}")

        # Progress reporting
        if z >= next_milestone:
            elapsed_time = time.time() - start_time
            print(f"Progress: {z/max_z*100:.1f}% ({z:,}/{max_z:,}) - Time: {elapsed_time:.1f}s")
            next_milestone += milestone

    print("\nAnalysis complete!")
    print("\nComplete family size distribution:")
    print("Size | Count | First Occurrence | Avg y_max/y_min | Avg Error from √2")
    print("-" * 70)

    for size in sorted(family_sizes.keys(), reverse=True):
        stats = family_stats[size]
        avg_ratio = stats['ratio_sum'] / stats['count']
        avg_error = stats['error_sum'] / stats['count']
        print(f"{size:4d} | {family_sizes[size]:5d} | z={first_occurrence[size]:10d} | {avg_ratio:13.8f} | {avg_error:.8f}")

    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")

    # Additional summary statistics
    total_families = sum(family_sizes.values())
    print(f"\nTotal number of families found: {total_families:,}")
    print(f"Number of different family sizes: {len(family_sizes)}")
    print(f"Smallest family size: {min(family_sizes.keys())}")
    print(f"Largest family size: {max(family_sizes.keys())}")

if __name__ == "__main__":
    main()
