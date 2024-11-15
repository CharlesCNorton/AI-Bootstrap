import numpy as np
from scipy.spatial import Voronoi, Delaunay
from ripser import ripser
from gudhi import SimplexTree

def generate_voronoi_diagram(num_points, dim):
    """ Creates a Delauanay triangulation from a Voronoi diagram based on random points. """
    points = np.random.rand(num_points, dim)
    vor = Voronoi(points)
    delau = Delaunay(points)
    return vor, delau

def compute_persistent_homology(simplices):
    """ Compute persistent homology of the Delaunay triangulation. """
    simplex_tree = SimplexTree()
    for i, simplex in enumerate(simplices):
        simplex_tree.insert(simplex, filtration=i)
    simplex_tree.initialize_filtration()
    simplex_tree.compute_persistence()
    return simplex_tree.persistence_intervals_in_dimension(0), simplex_tree.persistence_intervals_in_dimension(1)

def rigorous_invariant(persistence_intervals):
    """ Compute an invariant that is a geometric function of persistent features. """
    # Example invariant: Sum of squares of lifetimes
    return sum([(d[1] - d[0]) ** 2 for intervals in persistence_intervals for d in intervals if d[1] < np.inf])

def test_rigorous_invariant():
    num_tests = 10000
    success_count = 0
    failure_details = []

    for _ in range(num_tests):
        num_points = np.random.randint(20, 50)
        dim = 2  # Using 2D for Voronoi diagrams
        vor, delau = generate_voronoi_diagram(num_points, dim)

        # Get simplices from Delaunay triangulation for persistent homology computation
        persistence_intervals_0, persistence_intervals_1 = compute_persistent_homology(delau.simplices)

        complexity = len(persistence_intervals_0) + len(persistence_intervals_1)

        # Compute the invariant
        invariant_value = rigorous_invariant([persistence_intervals_0, persistence_intervals_1])

        # Verify if the invariant is providing an effective bound
        if invariant_value >= complexity:
            success_count += 1
        else:
            failure_details.append((num_points, complexity, invariant_value))

    print(f"Success Rate: {success_count / num_tests * 100}%")
    for detail in failure_details:
        print(f"Failed for points: {detail[0]}, complexity: {detail[1]}, invariant: {detail[2]:.2f}")

test_rigorous_invariant()
