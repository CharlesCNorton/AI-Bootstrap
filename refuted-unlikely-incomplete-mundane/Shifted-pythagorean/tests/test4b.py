from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict
from collections import defaultdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import time

@dataclass(frozen=True)
class Solution:
    x: int
    y: int
    z: int

    def verify(self) -> bool:
        return self.x*self.x + self.y*y == self.z*self.z + 1

def find_solutions_for_z(z: int) -> List[Tuple[int, int, int]]:
    """Find all solutions for a given z-value"""
    solutions = []
    z_squared_plus_1 = z*z + 1

    # Optimize the search range
    max_x = isqrt(z_squared_plus_1 - 1)

    for x in range(2, max_x + 1):
        y_squared = z_squared_plus_1 - x*x
        if y_squared > 0:
            y = isqrt(y_squared)
            if y*y == y_squared and y > x:
                solutions.append((x, y, z))

    return solutions

def massive_parallel_search(start: int, end: int, min_family_size: int = 20) -> Dict:
    """
    Aggressively search for large solution families
    Returns all families above min_family_size
    """
    large_families = {}
    chunk_size = 10000

    print(f"Starting massive search from {start:,} to {end:,}")
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            futures = [executor.submit(find_solutions_for_z, z)
                      for z in range(chunk_start, chunk_end)]

            for z, future in zip(range(chunk_start, chunk_end), futures):
                solutions = future.result()
                if len(solutions) >= min_family_size:
                    large_families[z] = solutions

                if z % 100000 == 0:
                    elapsed = time.time() - start_time
                    progress = (z - start) / (end - start) * 100
                    print(f"Progress: {progress:.2f}% | Z = {z:,} | Time: {elapsed:.2f}s")
                    print(f"Large families found: {len(large_families)}")
                    if large_families:
                        print(f"Largest so far: {max(len(sols) for sols in large_families.values())}")

    return large_families

def analyze_large_families(families: Dict[int, List[Tuple[int, int, int]]]) -> Dict:
    """Analyze patterns in large families"""
    analysis = {
        'size_distribution': defaultdict(int),
        'z_gaps': [],
        'size_progression': [],
        'largest_families': []
    }

    # Sort families by size
    sorted_families = sorted(
        [(z, len(sols)) for z, sols in families.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Analyze size distribution
    for z, size in sorted_families:
        analysis['size_distribution'][size] += 1

    # Analyze gaps between z-values with large families
    z_values = sorted(families.keys())
    analysis['z_gaps'] = [z_values[i+1] - z_values[i]
                         for i in range(len(z_values)-1)]

    # Track size progression
    analysis['size_progression'] = [size for _, size in sorted_families]

    # Detailed analysis of largest families
    top_10 = sorted_families[:10]
    for z, size in top_10:
        solutions = families[z]
        analysis['largest_families'].append({
            'z': z,
            'size': size,
            'x_range': (min(s[0] for s in solutions), max(s[0] for s in solutions)),
            'y_range': (min(s[1] for s in solutions), max(s[1] for s in solutions)),
            'ratio_range': (
                min(s[0]/s[1] for s in solutions),
                max(s[0]/s[1] for s in solutions)
            )
        })

    return analysis

def main():
    # Much more aggressive search
    START = 1_000
    END = 1_000_000  # One million
    MIN_FAMILY_SIZE = 20

    print(f"Starting aggressive search up to {END:,}")

    large_families = massive_parallel_search(START, END, MIN_FAMILY_SIZE)
    analysis = analyze_large_families(large_families)

    print("\n=== COMPREHENSIVE RESULTS ===")
    print(f"\nTotal large families found: {len(large_families)}")
    print("\nSize Distribution:")
    for size, count in sorted(analysis['size_distribution'].items(), reverse=True):
        print(f"Size {size}: {count} families")

    print("\nLargest Families:")
    for family in analysis['largest_families']:
        print(f"\nZ = {family['z']:,}")
        print(f"Size: {family['size']}")
        print(f"X range: {family['x_range']}")
        print(f"Y range: {family['y_range']}")
        print(f"Ratio range: ({family['ratio_range'][0]:.3f}, {family['ratio_range'][1]:.3f})")

    print("\nZ-value Gap Analysis:")
    gaps = np.array(analysis['z_gaps'])
    print(f"Min gap: {gaps.min():,}")
    print(f"Max gap: {gaps.max():,}")
    print(f"Mean gap: {gaps.mean():.2f}")
    print(f"Median gap: {np.median(gaps):.2f}")

if __name__ == "__main__":
    main()
