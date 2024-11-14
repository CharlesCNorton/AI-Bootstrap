import numpy as np
import gudhi as gd
from tqdm import tqdm

# Function to calculate K_M based on Betti numbers, vertices, and dimension
def calculate_k_m(betti_numbers, num_vertices, dimension, constant_factor=2.0, interaction_strength=0.3):
    """Calculate K_M using given parameters."""
    complexity = sum(betti_numbers)
    K_M = constant_factor * (complexity + interaction_strength * num_vertices * dimension)
    return K_M, complexity

# Function to check if K_M is an upper bound for complexity
def verify_bound(betti_numbers, num_vertices, dimension, constant_factor=2.0, interaction_strength=0.3):
    """Verify if K_M is an upper bound for the given complexity."""
    K_M, complexity = calculate_k_m(betti_numbers, num_vertices, dimension, constant_factor, interaction_strength)
    bound_check = K_M >= complexity
    return bound_check, K_M, complexity

# Function to generate Alpha Complex with GUDHI
def create_alpha_complex(points):
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.compute_persistence()
    return simplex_tree

# Function to generate points on the surface of a torus
def generate_torus_points(num_points, R=3, r=1):
    """Generate points on a torus with major radius R and minor radius r."""
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.vstack((x, y, z)).T

# Function to test specific known topologies with multiple filtrations
def test_topology(points, dimension, name):
    max_edge_lengths = [1.0, 2.0, 4.0, 6.0]  # Using multiple edge lengths to capture features
    betti_numbers_aggregated = []

    print(f"Testing {name}:")

    for max_edge_length in max_edge_lengths:
        rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)

        # Compute persistence
        simplex_tree.compute_persistence()
        betti_numbers = simplex_tree.betti_numbers()
        betti_numbers_aggregated.append(betti_numbers)
        print(f"Max Edge Length {max_edge_length} -> Betti Numbers: {betti_numbers}")

    # Summarizing the Betti numbers over multiple filtrations
    max_betti_numbers = np.max(betti_numbers_aggregated, axis=0)
    num_vertices = len(points)
    bound_check, K_M, complexity = verify_bound(max_betti_numbers, num_vertices, dimension)

    print(f"{name} Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}")
    print(f"Aggregated Betti Numbers: {max_betti_numbers}\n")

# Test known simplicial complexes with GUDHI
print("Running tests on well-known topologies using GUDHI with dynamic filtration...")

# 1. S^3 (3-dimensional sphere)
points_sphere_3d = np.random.normal(size=(8000, 4))  # Using Gaussian distribution for better spread in 4D with 8,000 points
test_topology(points_sphere_3d, dimension=3, name="S^3 (3-dimensional sphere)")

# 2. T^2 (2-dimensional torus)
points_torus_2d = generate_torus_points(10000)  # Using 10,000 points on the torus
test_topology(points_torus_2d, dimension=2, name="T^2 (2-dimensional torus)")

# 3. Klein Bottle
points_klein_bottle = np.random.rand(8000, 4)  # Using 8,000 points in 4 dimensions
test_topology(points_klein_bottle, dimension=2, name="Klein Bottle")

# 4. RP^2 (Projective Plane)
points_rp2 = np.random.normal(size=(8000, 3))  # Better coverage in 3D with 8,000 points
test_topology(points_rp2, dimension=2, name="RP^2 (Projective Plane)")

# 5. Möbius Strip
points_mobius = np.random.rand(8000, 3)  # 8,000 points for better representation of Möbius strip
test_topology(points_mobius, dimension=2, name="Möbius Strip")

# 6. S^4 (4-dimensional sphere)
points_sphere_4d = np.random.normal(size=(10000, 5))  # Using 10,000 points in 5D
test_topology(points_sphere_4d, dimension=4, name="S^4 (4-dimensional sphere)")
