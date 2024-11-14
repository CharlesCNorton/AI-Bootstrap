import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from gudhi import SimplexTree
from tqdm import tqdm

def generate_voronoi_diagram(num_points, dim):
    """Creates a Delaunay triangulation based on random points for persistent homology."""
    points = np.random.rand(num_points, dim)
    return points

def compute_persistent_homology(points):
    """Compute persistent homology from a Delaunay triangulation of points."""
    delaunay = Delaunay(points)
    simplex_tree = SimplexTree()

    # Insert each simplex from the Delaunay triangulation into the simplex tree
    for simplex in delaunay.simplices:
        simplex_tree.insert(simplex, filtration=len(simplex))

    # Initialize filtration and compute persistence
    simplex_tree.initialize_filtration()
    simplex_tree.compute_persistence()

    # Return persistence intervals for dimensions 0 and 1
    return (
        simplex_tree.persistence_intervals_in_dimension(0),
        simplex_tree.persistence_intervals_in_dimension(1)
    )

def rigorous_invariant(persistence_intervals, num_points):
    """Compute an invariant with cross-terms, logarithmic scaling, and adaptive base."""
    # Sum of squares of lifetimes for each interval
    base_invariant = sum([(interval[1] - interval[0]) ** 2 for dim_intervals in persistence_intervals for interval in dim_intervals if interval[1] < np.inf])

    # Cross-term for interaction between low-dimensional features
    cross_term = 0
    if len(persistence_intervals[0]) > 1 and len(persistence_intervals[1]) > 1:
        for interval_0, interval_1 in zip(persistence_intervals[0], persistence_intervals[1]):
            cross_term += abs((interval_0[1] - interval_0[0]) * (interval_1[1] - interval_1[0]))

    # Logarithmic scaling for a minimum bound, scaled with point count
    log_term = np.log1p(num_points)

    # Combining terms into a refined invariant
    refined_invariant = base_invariant + cross_term + log_term

    return refined_invariant

def simulate_complexes_with_refined_invariant():
    """Simulate complexes with the refined invariant and collect statistics."""
    num_tests = 100000  # Number of tests per batch
    results = []

    for _ in tqdm(range(num_tests)):
        # Random number of points and dimension for each test
        num_points = np.random.randint(10, 30)
        dim = np.random.randint(2, 4)

        # Generate points and compute persistent homology
        points = generate_voronoi_diagram(num_points, dim)
        persistence_intervals_0, persistence_intervals_1 = compute_persistent_homology(points)

        # Compute complexity as sum of Betti numbers
        complexity = len(persistence_intervals_0) + len(persistence_intervals_1)

        # Refined invariant calculation
        refined_invariant = rigorous_invariant([persistence_intervals_0, persistence_intervals_1], num_points)

        # Check if the refined invariant provides an upper bound
        bound_check = refined_invariant >= complexity
        results.append({
            "num_points": num_points,
            "dimension": dim,
            "complexity": complexity,
            "refined_invariant": refined_invariant,
            "bound_check": bound_check
        })

    # Create a DataFrame to analyze results
    df = pd.DataFrame(results)
    return df

# Run the refined simulation and display results
refined_results_df = simulate_complexes_with_refined_invariant()
success_rate = refined_results_df["bound_check"].mean() * 100
print(f"Success Rate with Refined Invariant: {success_rate:.2f}%")
print(refined_results_df[~refined_results_df["bound_check"]])  # Display any failures for further analysis
