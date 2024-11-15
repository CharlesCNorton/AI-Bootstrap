import numpy as np
from collections import Counter
from math import isqrt, sqrt

# Helper function to generate lattice points within a given radius in 3D
def generate_3d_lattice_points(radius):
    points = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                if x**2 + y**2 + z**2 <= radius**2:
                    points.append((x, y, z))
    return points

# Test 1: Validate Simple Geometric Configurations (family sizes 23, 31)
def test_simple_geometries():
    radius = 50  # Set an appropriate radius
    points = generate_3d_lattice_points(radius)
    distances = [sqrt(x**2 + y**2 + z**2) for x, y, z in points]
    simple_distances = [1.0, 2.0]
    result = Counter([round(d, 3) for d in distances if round(d, 3) in simple_distances])

    print("Simple Geometric Configurations Test Results:")
    print(result)

# Test 2: Validate Complex Geometric Configurations (family sizes 95, 215)
def test_complex_geometries():
    radius = 100  # Set larger radius for complex structures
    points = generate_3d_lattice_points(radius)
    distances = [sqrt(x**2 + y**2 + z**2) for x, y, z in points]
    complex_distances = [59.21, 59.72, 58.73]  # Most frequent complex distances
    result = Counter([round(d, 3) for d in distances if round(d, 3) in complex_distances])

    print("Complex Geometric Configurations Test Results:")
    print(result)

# Test 3: Check Convergence Toward sqrt(2) as family size increases
def test_convergence_sqrt2():
    z_values = [1000, 10000, 100000, 1000000]  # Example z values for varying family sizes
    sqrt2 = sqrt(2)
    for z in z_values:
        solutions = []
        for x in range(2, z):
            y_squared = z*z + 1 - x*x
            if y_squared > 0:
                y = isqrt(y_squared)
                if y*y == y_squared and y > x:
                    solutions.append((x, y))

        if solutions:
            y_max = max([y for _, y in solutions])
            y_min = min([y for _, y in solutions])
            ratio = y_max / y_min
            print(f"z = {z}, y_max/y_min = {ratio:.6f}, Error from sqrt(2) = {abs(ratio - sqrt2):.6f}")
        else:
            print(f"z = {z}, No solutions found")

# Test 4: Analyze Family Size Thresholds (sharp drop-off beyond size 95)
def test_family_size_thresholds():
    z_values = range(2, 1000, 100)  # Test for z values to examine family size growth
    threshold_family_size = 95
    families_above_threshold = []

    for z in z_values:
        solutions = []
        for x in range(2, z):
            y_squared = z*z + 1 - x*x
            if y_squared > 0:
                y = isqrt(y_squared)
                if y*y == y_squared and y > x:
                    solutions.append((x, y))
        family_size = len(solutions)
        if family_size > threshold_family_size:
            families_above_threshold.append((z, family_size))

    print("Family Sizes Above Threshold (95 Solutions):")
    for z, size in families_above_threshold:
        print(f"z = {z}, Family Size = {size}")

# Test 5: Distribution of Unique Distances as Radius Increases
def test_unique_distance_distribution():
    radii = [10, 20, 30, 40, 50]  # Test across increasing radii
    for r in radii:
        points = generate_3d_lattice_points(r)
        distances = [sqrt(x**2 + y**2 + z**2) for x, y, z in points]
        unique_distances = sorted(set([round(d, 3) for d in distances]))
        print(f"Radius = {r}, Unique Distances = {len(unique_distances)}")

# Execute all tests
test_simple_geometries()
test_complex_geometries()
test_convergence_sqrt2()
test_family_size_thresholds()
test_unique_distance_distribution()
