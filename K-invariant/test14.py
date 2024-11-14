import numpy as np

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

# Test known simplicial complexes

# 1. n-dimensional Sphere S^n
# Betti numbers for S^n: [1, 0, ..., 0, 1] (length n+1)
print("Testing S^3 (3-dimensional sphere):")
betti_numbers_sphere = [1, 0, 0, 1]  # Betti numbers for S^3
num_vertices_sphere = 5  # Minimal representation as a simplicial complex
dimension_sphere = 3

bound_check, K_M, complexity = verify_bound(betti_numbers_sphere, num_vertices_sphere, dimension_sphere)
print(f"S^3 Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")

# 2. Torus T^2
# Betti numbers for T^2: [1, 2, 1] (0th, 1st, and 2nd homology groups)
print("Testing T^2 (2-dimensional torus):")
betti_numbers_torus = [1, 2, 1]  # Betti numbers for T^2
num_vertices_torus = 7  # Approximate representation as a simplicial complex
dimension_torus = 2

bound_check, K_M, complexity = verify_bound(betti_numbers_torus, num_vertices_torus, dimension_torus)
print(f"T^2 Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")

# 3. Klein Bottle
# Betti numbers for the Klein bottle: [1, 1, 0]
print("Testing Klein Bottle:")
betti_numbers_klein = [1, 1, 0]  # Betti numbers for the Klein bottle
num_vertices_klein = 8  # Approximate representation as a simplicial complex
dimension_klein = 2

bound_check, K_M, complexity = verify_bound(betti_numbers_klein, num_vertices_klein, dimension_klein)
print(f"Klein Bottle Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")

# 4. Projective Plane RP^2
# Betti numbers for RP^2: [1, 0, 1]
print("Testing RP^2 (Projective Plane):")
betti_numbers_rp2 = [1, 0, 1]  # Betti numbers for RP^2
num_vertices_rp2 = 6  # Approximate representation as a simplicial complex
dimension_rp2 = 2

bound_check, K_M, complexity = verify_bound(betti_numbers_rp2, num_vertices_rp2, dimension_rp2)
print(f"RP^2 Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")

# 5. Möbius Strip
# Betti numbers for Möbius strip: [1, 1]
print("Testing Möbius Strip:")
betti_numbers_mobius = [1, 1]  # Betti numbers for the Möbius strip
num_vertices_mobius = 5  # Approximate representation as a simplicial complex
dimension_mobius = 2

bound_check, K_M, complexity = verify_bound(betti_numbers_mobius, num_vertices_mobius, dimension_mobius)
print(f"Möbius Strip Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")

# 6. 4-Dimensional Sphere S^4
# Betti numbers for S^4: [1, 0, 0, 0, 1]
print("Testing S^4 (4-dimensional sphere):")
betti_numbers_sphere4 = [1, 0, 0, 0, 1]  # Betti numbers for S^4
num_vertices_sphere4 = 9  # Minimal representation as a simplicial complex
dimension_sphere4 = 4

bound_check, K_M, complexity = verify_bound(betti_numbers_sphere4, num_vertices_sphere4, dimension_sphere4)
print(f"S^4 Bound Check: {bound_check}, K_M: {K_M}, Complexity: {complexity}\n")
