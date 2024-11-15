from math import isqrt, sqrt
from typing import List, Tuple
from collections import defaultdict

def verify_solution(x: int, y: int, z: int) -> bool:
    return x * x + y * y == z * z + 1

def find_solutions(z: int) -> List[Tuple[int, int]]:
    solutions = []
    for x in range(2, z):
        y_squared = z * z + 1 - x * x
        if y_squared > 0:
            y = isqrt(y_squared)
            if y * y == y_squared and y > x:
                solutions.append((x, y))
    return sorted(solutions)

def analyze_ratio(solutions: List[Tuple[int, int]]) -> float:
    if not solutions:
        return 0
    y_values = [y for _, y in solutions]
    return max(y_values) / min(y_values)

def verify_hierarchy(start_z: int, end_z: int, min_solutions: int = 20) -> dict:
    """Verify the hierarchy of family sizes"""
    family_sizes = defaultdict(int)

    for z in range(start_z, end_z + 1):
        solutions = find_solutions(z)
        if len(solutions) >= min_solutions:
            family_sizes[len(solutions)] += 1

        if z % 10000 == 0:
            print(f"Progress: {z/end_z*100:.1f}%")

    return dict(family_sizes)

# First verify the 95-solution case
z = 330182
solutions = find_solutions(z)
ratio = analyze_ratio(solutions)
print("\nVerification of first 95-solution family:")
print({
    "Solutions found": len(solutions),
    "y_max/y_min ratio": ratio,
    "Error from √2": abs(ratio - sqrt(2))
})

# Then verify the hierarchy (this will take longer)
print("\nVerifying solution hierarchy up to 100,000...")
hierarchy = verify_hierarchy(1, 100000)
print("\nFamily size distribution:")
for size in sorted(hierarchy.keys(), reverse=True):
    print(f"Size {size}: {hierarchy[size]} families")
