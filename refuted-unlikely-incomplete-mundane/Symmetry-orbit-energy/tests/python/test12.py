import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt

def compute_lie_group_negentropy(group_type, rank, num_points=1000):
    """
    Compute the negentropy (-S) for different Lie groups

    Parameters:
    group_type: str ('classical' or 'exceptional')
    rank: int (rank of the Lie group)

    Returns:
    float: computed negentropy value
    """

    def density_function(x, y, z):
        """
        Density function for conjugacy classes
        """
        if group_type == 'classical':
            # Classical groups have more uniform distribution
            return np.exp(-(x**2 + y**2 + z**2)/(2*rank))
        else:
            # Exceptional groups have more concentrated distribution
            return np.exp(-(x**2 + y**2 + z**2)/(rank)) * (1 + 0.5*np.cos(np.sqrt(x**2 + y**2 + z**2)))

    def integrand(x, y, z):
        """
        Integrand for entropy calculation: f(x) log(f(x))
        Note: We return positive value since we're computing negentropy
        """
        f = density_function(x, y, z)
        return f * np.log(f) if f > 0 else 0

    # Integration limits
    limits = [[-5, 5], [-5, 5], [-5, 5]]

    # Normalize density function
    norm, _ = nquad(density_function, limits)

    # Compute negentropy using numerical integration
    negentropy, _ = nquad(integrand, limits)

    # Scale by normalization
    negentropy /= norm

    return negentropy

def run_negentropy_comparison_test():
    """
    Run test comparing classical and exceptional Lie group negentropies
    """
    ranks = range(2, 9)
    classical_negentropies = []
    exceptional_negentropies = []

    for r in ranks:
        classical_neg = compute_lie_group_negentropy('classical', r)
        exceptional_neg = compute_lie_group_negentropy('exceptional', r)

        classical_negentropies.append(classical_neg)
        exceptional_negentropies.append(exceptional_neg)

        print(f"Rank {r}:")
        print(f"Classical Group Negentropy: {classical_neg:.4f}")
        print(f"Exceptional Group Negentropy: {exceptional_neg:.4f}")
        print(f"Difference: {exceptional_neg - classical_neg:.4f}\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, classical_negentropies, 'b-', label='Classical Groups')
    plt.plot(ranks, exceptional_negentropies, 'r-', label='Exceptional Groups')
    plt.xlabel('Rank')
    plt.ylabel('Negentropy')
    plt.title('Negentropy Comparison: Classical vs Exceptional Lie Groups')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Statistical test
    negentropy_differences = np.array(exceptional_negentropies) - np.array(classical_negentropies)
    mean_difference = np.mean(negentropy_differences)
    std_difference = np.std(negentropy_differences)
    t_statistic = mean_difference / (std_difference / np.sqrt(len(ranks)))

    return {
        'classical_negentropies': classical_negentropies,
        'exceptional_negentropies': exceptional_negentropies,
        'mean_difference': mean_difference,
        't_statistic': t_statistic
    }

# Run the test
results = run_negentropy_comparison_test()
