import numpy as np
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mpmath

# Generate synthetic derivative data for persistent homology
def generate_derivative_vector_space(zeros, max_order):
    """
    Generate a synthetic derivative vector space for testing persistent homology.
    Parameters:
        zeros (list): Nontrivial zeros of the Riemann zeta function.
        max_order (int): Maximum order of derivatives to include.
    Returns:
        numpy.ndarray: Matrix where rows are zeros and columns are derivative orders.
    """
    # Simulate normalized derivatives as synthetic data
    vector_space = np.array([
        [(zero ** (n - 0.5)) for n in range(1, max_order + 1)]
        for zero in zeros
    ])
    return vector_space

# Load or simulate zeros of the Riemann zeta function
def load_riemann_zeros(num_zeros):
    """
    Simulate or load nontrivial zeros of the Riemann zeta function.
    Parameters:
        num_zeros (int): Number of zeros to include.
    Returns:
        numpy.ndarray: Array of zeros.
    """
    return np.array([mpmath.im(mpmath.zetazero(n)) for n in range(1, num_zeros + 1)])

# Normalize the derivative vector space
def normalize_vector_space(vector_space):
    """
    Normalize the derivative vector space to mitigate scaling artifacts.
    Parameters:
        vector_space (numpy.ndarray): Raw derivative vector space.
    Returns:
        numpy.ndarray: Normalized vector space.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(vector_space)

# Persistent Homology Analysis
def analyze_persistent_homology(vector_space):
    """
    Analyze persistent homology of the given derivative vector space.
    Parameters:
        vector_space (numpy.ndarray): Derivative vector space.
    """
    # Normalize vector space and compute distances
    normalized_space = normalize_vector_space(vector_space)
    distances = euclidean_distances(normalized_space)

    # Optional: Add small noise to avoid degenerate features
    distances += np.random.normal(0, 1e-6, distances.shape)

    # Compute persistent homology using Ripser
    diagrams = ripser(distances, distance_matrix=True)['dgms']

    # Plot persistence diagrams
    plot_diagrams(diagrams, show=True)

    # Extract birth-death pairs and compute ratios (filter out zero-birth features)
    birth_death_ratios = []
    for dgm in diagrams:
        for birth, death in dgm:
            if birth > 0 and death < np.inf:  # Ignore zero-birth and infinite-death features
                birth_death_ratios.append(death / birth)

    # Output summary of ratios
    conjectural_scaling = np.exp(np.pi / np.log(2))
    close_to_conjecture = [abs(ratio - conjectural_scaling) < 0.1 for ratio in birth_death_ratios]

    print("Birth-Death Ratios:")
    print(birth_death_ratios)
    print("\nClose to Conjectural Scaling (±0.1):", close_to_conjecture)

# Main Function
def main():
    # Number of zeros and maximum derivative order
    num_zeros = 20
    max_order = 10

    # Generate data
    zeros = load_riemann_zeros(num_zeros)
    vector_space = generate_derivative_vector_space(zeros, max_order)

    # Analyze persistent homology
    analyze_persistent_homology(vector_space)

if __name__ == "__main__":
    main()
