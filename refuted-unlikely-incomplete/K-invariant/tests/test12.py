import snappy
import numpy as np
import gudhi
import itertools

# Step 1: Generate a List of Knot Complements Using SnapPy
def generate_knot_complements(knot_names):
    manifolds = [snappy.Manifold(knot) for knot in knot_names]
    return manifolds

# Step 2: Compute Hyperbolic Invariants for Validation
def compute_hyperbolic_invariants(manifold):
    try:
        volume = manifold.volume()
        cusp_invariants = manifold.cusp_info('complete')
        return volume, cusp_invariants
    except RuntimeError as e:
        return None, None

# Step 3: Compute Your Invariant for the Knot Complement (as a 3-manifold)
def compute_my_invariant(manifold):
    simplex_tree = gudhi.SimplexTree()

    # Extract triangulation information from SnapPy
    try:
        # Get the tetrahedra shapes from SnapPy.
        tetrahedra = manifold.tetrahedra_shapes('rect')  # Getting the tetrahedra shapes (complex shapes in rectangular form)

        # Each tetrahedron can be represented as a 3-simplex with 4 vertices.
        vertex_count = 0
        for tetra in tetrahedra:
            simplex = [vertex_count, vertex_count + 1, vertex_count + 2, vertex_count + 3]
            simplex_tree.insert(simplex)
            vertex_count += 4

        # Assign filtration values (dummy values in this case)
        # We assign the filtration value manually for each simplex inserted
        for filtration_value, (simplex, _) in enumerate(simplex_tree.get_filtration()):
            simplex_tree.assign_filtration(simplex, filtration_value)

    except Exception as e:
        print(f"Error processing manifold {manifold.name()}: {e}")
        return None

    # Compute persistence to extract features from simplicial complex
    try:
        simplex_tree.compute_persistence()
    except Exception as e:
        print(f"Error computing persistence for {manifold.name()}: {e}")
        return None

    betti_numbers = simplex_tree.betti_numbers()

    # Calculate invariant using Betti numbers and persistence features
    weighted_betti_sum = sum((i + 1) * b for i, b in enumerate(betti_numbers))

    # Compute total persistence lifetime
    persistence_pairs = simplex_tree.persistence()

    # Corrected way to access the birth and death values
    total_lifetime = sum(death - birth for _, (birth, death) in persistence_pairs if death != float('inf'))

    # Convert the filtration to a list to determine the number of simplices
    filtration_list = list(simplex_tree.get_filtration())
    num_simplices = len(filtration_list)

    # Combine metrics to form the invariant
    if num_simplices > 0:
        normalized_betti = weighted_betti_sum / num_simplices
    else:
        normalized_betti = 0

    # Final invariant is a combination of different metrics
    invariant = 0.5 * normalized_betti + 0.3 * total_lifetime + 0.2 * num_simplices

    return invariant

# Step 4: Compare Knot Complements Using Your Invariant
def classify_complements(manifolds):
    results = []
    for (m1, m2) in itertools.combinations(manifolds, 2):
        inv_1 = compute_my_invariant(m1)
        inv_2 = compute_my_invariant(m2)

        if inv_1 is None or inv_2 is None:
            continue  # Skip comparison if invariant calculation failed

        # Are the invariants close enough to suggest homeomorphism?
        if abs(inv_1 - inv_2) < 0.01:  # Assuming some error threshold
            results.append((m1.name(), m2.name(), 'Equivalent'))
        else:
            results.append((m1.name(), m2.name(), 'Distinct'))

    return results

# Main Execution
if __name__ == "__main__":
    # Example set of prime knots; you can expand this list
    knot_names = ['3_1', '4_1', '5_2', '6_3', '7_1', '8_19', '9_42']
    manifolds = generate_knot_complements(knot_names)

    classification_results = classify_complements(manifolds)

    # Print Results
    for knot_1, knot_2, relation in classification_results:
        print(f"Knot Complement Comparison: {knot_1} and {knot_2} are {relation}.")
