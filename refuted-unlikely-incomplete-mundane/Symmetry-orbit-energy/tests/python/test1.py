import numpy as np
from scipy.stats import unitary_group, special_ortho_group
from scipy.linalg import logm
import matplotlib.pyplot as plt

def haar_measure_su_n(n, num_samples):
    """
    Generates random elements from the SU(n) group using the Haar measure.
    """
    return [unitary_group.rvs(n) for _ in range(num_samples)]

def haar_measure_so_n(n, num_samples):
    """
    Generates random elements from the SO(n) group using the Haar measure.
    """
    return [special_ortho_group.rvs(n) for _ in range(num_samples)]

def calculate_entropy(samples, n_bins=100):
    """
    Calculates the entropy based on the distribution of eigenvalues of the samples.
    """
    eigenvalue_lists = []
    for matrix in samples:
        eigenvalues = np.linalg.eigvals(matrix)
        phases = np.angle(eigenvalues)
        eigenvalue_lists.extend(phases)

    hist, bin_edges = np.histogram(eigenvalue_lists, bins=n_bins, density=True)
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))  # Small term to avoid log(0)
    return entropy

def test_symmetry_orbit_entropy(group_type='SU', n=2, num_samples=1000, n_bins=100):
    """
    Tests the Symmetry Orbit Entropy for a given Lie group.
    """
    if group_type == 'SU':
        samples = haar_measure_su_n(n, num_samples)
    elif group_type == 'SO':
        samples = haar_measure_so_n(n, num_samples)
    else:
        raise ValueError("Unsupported group type. Choose 'SU' or 'SO'.")

    entropy = calculate_entropy(samples, n_bins=n_bins)
    print(f"Symmetry Orbit Entropy for {group_type}({n}): {entropy}")

    # Optional: Plot the histogram of eigenvalue phases
    eigenvalue_phases = []
    for matrix in samples:
        eigenvalues = np.linalg.eigvals(matrix)
        phases = np.angle(eigenvalues)
        eigenvalue_phases.extend(phases)

    plt.hist(eigenvalue_phases, bins=n_bins, density=True)
    plt.title(f"Distribution of Eigenvalue Phases for {group_type}({n})")
    plt.xlabel("Phase")
    plt.ylabel("Density")
    plt.show()

# Testing the entropy for SU(2), SU(3), and SU(4)
test_symmetry_orbit_entropy(group_type='SU', n=2, num_samples=5000, n_bins=100)
test_symmetry_orbit_entropy(group_type='SU', n=3, num_samples=5000, n_bins=100)
test_symmetry_orbit_entropy(group_type='SU', n=4, num_samples=5000, n_bins=100)

# Testing the entropy for SO(3), SO(4), and SO(5)
test_symmetry_orbit_entropy(group_type='SO', n=3, num_samples=5000, n_bins=100)
test_symmetry_orbit_entropy(group_type='SO', n=4, num_samples=5000, n_bins=100)
test_symmetry_orbit_entropy(group_type='SO', n=5, num_samples=5000, n_bins=100)
