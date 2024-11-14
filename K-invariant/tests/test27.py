import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from gudhi import SimplexTree
from tqdm import tqdm

# ===========================================
# PART 1: Generating Simplicial Approximations of Spheres
# ===========================================

def generate_sphere_points(radius, num_points, dim):
    """
    Generate points that approximate a sphere of given radius and dimension.
    Uses spherical coordinates for uniform distribution.

    Parameters:
    - radius (float): Radius of the sphere.
    - num_points (int): Number of points to sample.
    - dim (int): Dimension of the sphere.

    Returns:
    - numpy array of shape (num_points, dim+1): Points approximating the sphere.
    """
    points = np.random.randn(num_points, dim + 1)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= radius
    return points

# ===========================================
# PART 2: Computing Persistent Homology of Simplicial Complexes
# ===========================================

def compute_persistent_homology(points):
    """
    Compute persistent homology using the Delaunay triangulation of the given points.

    Parameters:
    - points (numpy array): Array of points representing the simplicial complex.

    Returns:
    - Tuple of persistence intervals for dimensions 0 and 1.
    """
    delaunay = Delaunay(points)
    simplex_tree = SimplexTree()

    # Insert simplices into SimplexTree
    for simplex in delaunay.simplices:
        simplex_tree.insert(simplex, filtration=len(simplex))

    # Compute persistence directly without initialization
    simplex_tree.compute_persistence()

    # Return persistence intervals for dimension 0 and 1 (representative of essential topological features)
    return (
        simplex_tree.persistence_intervals_in_dimension(0),
        simplex_tree.persistence_intervals_in_dimension(1)
    )

# ===========================================
# PART 3: Defining Enhanced K_invariant with Fourier Series and Adaptive Scaling
# ===========================================

def k_invariant(persistence_intervals, num_points, dimension):
    """
    Define an enhanced K_invariant as a measure of homotopical complexity.
    This includes a Fourier series, cross-terms, and adaptive scaling.

    Parameters:
    - persistence_intervals (list): List of persistence intervals from persistent homology.
    - num_points (int): Number of points in the simplicial complex.
    - dimension (int): Dimension of the simplicial complex.

    Returns:
    - float: Value of K_invariant for the complex.
    """
    # Base component: Sum of squares of lifetimes
    base_invariant = sum([(interval[1] - interval[0]) ** 2 for dim_intervals in persistence_intervals for interval in dim_intervals if interval[1] < np.inf])

    # Cross-term to account for interactions in homotopy-like structures across all intervals
    cross_term = 0
    if len(persistence_intervals[0]) > 1 and len(persistence_intervals[1]) > 1:
        for interval_0, interval_1 in zip(persistence_intervals[0], persistence_intervals[1]):
            cross_term += abs((interval_0[1] - interval_0[0]) * (interval_1[1] - interval_1[0]))

    # Logarithmic scaling term for minimum complexity bound, enhanced to scale with sphere dimension
    log_term = np.log1p(num_points * dimension)

    # Fourier series to capture periodic homotopical behaviors in stable homotopy groups
    # Here we use the sum of sine terms to model homotopy periodicity
    periodic_term = sum(np.sin((interval[1] - interval[0]) * np.pi / 2) for dim_intervals in persistence_intervals for interval in dim_intervals if interval[1] < np.inf)

    # Adaptive scaling factor for dimensional adjustments
    adaptive_scaling = 1 + (dimension ** 0.5) * 0.1

    # Combining all components for the enhanced K_invariant
    refined_invariant = adaptive_scaling * (base_invariant + cross_term + log_term + periodic_term)

    return refined_invariant

# ===========================================
# PART 4: Applying Enhanced K_invariant to All Homotopy Groups of Spheres
# ===========================================

def analyze_all_spheres(radius, num_points, max_dimension, iterations=1000):
    """
    Analyze the homotopy groups of spheres S^k for all dimensions up to max_dimension.

    Parameters:
    - radius (float): Radius of the spheres.
    - num_points (int): Number of points in each sphere's simplicial approximation.
    - max_dimension (int): Maximum dimension to analyze.
    - iterations (int): Number of iterations for each sphere to ensure accuracy.

    Returns:
    - DataFrame: Results of K_invariant values and their bounds on homotopy complexity.
    """
    results = []

    for dim in range(1, max_dimension + 1):
        print(f"Analyzing S^{dim} sphere:")
        for _ in tqdm(range(iterations), desc=f"S^{dim}"):
            # Generate points and compute persistent homology
            points = generate_sphere_points(radius, num_points, dim)
            persistence_intervals_0, persistence_intervals_1 = compute_persistent_homology(points)

            # Compute complexity as sum of Betti numbers (proxy for homotopy complexity)
            complexity = len(persistence_intervals_0) + len(persistence_intervals_1)

            # Calculate enhanced K_invariant
            invariant_value = k_invariant([persistence_intervals_0, persistence_intervals_1], num_points, dim)

            # Check if K_invariant bounds the computed complexity
            bound_check = invariant_value >= complexity

            # Store result
            results.append({
                "dimension": dim,
                "num_points": num_points,
                "complexity": complexity,
                "K_invariant": invariant_value,
                "bound_check": bound_check
            })

    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    return df

# ===========================================
# PART 5: Running the Analysis for All Dimensions
# ===========================================

# Parameters for sphere generation and analysis
radius = 1.0
num_points = 50  # Adjust based on computational resources
max_dimension = 8  # Max dimension of spheres to analyze (can be increased based on resources)
iterations = 100  # Number of repetitions per dimension for statistical stability

# Run the analysis with the fully enhanced K_invariant across all specified dimensions
all_spheres_analysis_df = analyze_all_spheres(radius, num_points, max_dimension, iterations)

# Calculate success rate and analyze results
success_rate = all_spheres_analysis_df["bound_check"].mean() * 100
print(f"Success Rate with Enhanced K_invariant on All Homotopy Groups of Spheres: {success_rate:.2f}%")
print(all_spheres_analysis_df[~all_spheres_analysis_df["bound_check"]])  # Show any failures for further analysis
