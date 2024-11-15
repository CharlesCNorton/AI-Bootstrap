import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def create_path_space(x, y, dim):
    """Create path space matrix as defined in the paper"""
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * dim)
    perturbation = np.random.uniform(-1, 1, (dim, dim))
    return np.eye(dim) + epsilon * np.exp(-0.3 * dist) * perturbation

def measure_properties(dim, samples=1000):
    """Measure reflexivity, symmetry, and transitivity with proper normalization"""
    R_values = []
    S_values = []
    T_values = []

    for _ in range(samples):
        x = np.random.uniform(-1, 1, dim)
        y = np.random.uniform(-1, 1, dim)
        z = np.random.uniform(-1, 1, dim)

        # Reflexivity with proper normalization
        P_xx = create_path_space(x, x, dim)
        R = 1 - (np.linalg.norm(P_xx - np.eye(dim), 'fro') /
                 np.linalg.norm(np.eye(dim), 'fro'))

        # Symmetry with proper normalization
        P_xy = create_path_space(x, y, dim)
        P_yx = create_path_space(y, x, dim)
        S = 1 - (np.linalg.norm(P_xy - P_yx.T, 'fro') /
                 (np.linalg.norm(P_xy, 'fro') + np.linalg.norm(P_yx, 'fro')))

        # Transitivity with proper normalization
        P_yz = create_path_space(y, z, dim)
        P_xz = create_path_space(x, z, dim)
        composition = P_xy @ P_yz
        T = 1 - (np.linalg.norm(composition - P_xz, 'fro') /
                 (np.linalg.norm(composition, 'fro') + np.linalg.norm(P_xz, 'fro')))

        R_values.append(R)
        S_values.append(S)
        T_values.append(T)

    return np.mean(R_values), np.mean(S_values), np.mean(T_values)

def analyze_properties():
    dimensions = range(1, 31)
    R_results = []
    S_results = []
    T_results = []

    for d in dimensions:
        print(f"Testing dimension {d}")
        R, S, T = measure_properties(d)
        R_results.append(R)
        S_results.append(S)
        T_results.append(T)

    # Test hierarchical stability
    hierarchy_valid = all(S_results[i] > T_results[i] > R_results[i]
                         for i in range(len(dimensions)))

    # Calculate ratio evolution
    S_R_ratios = [S/R for S, R in zip(S_results, R_results)]
    ratio_slope = np.polyfit(dimensions, S_R_ratios, 1)[0]

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(dimensions, R_results, 'b-', label='Reflexivity')
    plt.plot(dimensions, S_results, 'r-', label='Symmetry')
    plt.plot(dimensions, T_results, 'g-', label='Transitivity')
    plt.xlabel('Dimension')
    plt.ylabel('Property Value')
    plt.title('Path Space Properties vs Dimension')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate asymptotic values (using last 5 dimensions)
    R_asymp = np.mean(R_results[-5:])
    S_asymp = np.mean(S_results[-5:])
    T_asymp = np.mean(T_results[-5:])

    return {
        'hierarchy_valid': hierarchy_valid,
        'ratio_slope': ratio_slope,
        'R_asymptotic': R_asymp,
        'S_asymptotic': S_asymp,
        'T_asymptotic': T_asymp,
        'data': (R_results, S_results, T_results)
    }

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    results = analyze_properties()

    print("\nResults:")
    print(f"Hierarchy S > T > R valid: {results['hierarchy_valid']}")
    print(f"S/R ratio slope: {results['ratio_slope']:.6f}")
    print(f"\nAsymptotic values:")
    print(f"R_∞ ≈ {results['R_asymptotic']:.6f}")
    print(f"S_∞ ≈ {results['S_asymptotic']:.6f}")
    print(f"T_∞ ≈ {results['T_asymptotic']:.6f}")
