import numpy as np
from gudhi import RipsComplex
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_hopf_points(num_points=100, noise=0.0):
    """
    Generate points from the Hopf fibration S³ → S² with optional noise
    Returns both S³ points and their S² projections
    """
    # Generate points on S³
    points_s3 = np.random.normal(0, 1, (num_points, 4))
    points_s3 = points_s3 / np.linalg.norm(points_s3, axis=1)[:, np.newaxis]

    if noise > 0:
        points_s3 += np.random.normal(0, noise, points_s3.shape)
        points_s3 = points_s3 / np.linalg.norm(points_s3, axis=1)[:, np.newaxis]

    # Project to S² using Hopf map
    points_s2 = np.zeros((num_points, 3))
    for i in range(num_points):
        z1, z2 = points_s3[i, 0] + 1j*points_s3[i, 1], points_s3[i, 2] + 1j*points_s3[i, 3]
        points_s2[i, 0] = 2 * (z1 * np.conj(z2)).real
        points_s2[i, 1] = 2 * (z1 * np.conj(z2)).imag
        points_s2[i, 2] = (abs(z1)**2 - abs(z2)**2)

    return points_s3, points_s2

def compute_enhanced_k_invariant(points, max_dim):
    """Enhanced K-invariant with all components from Appendix D.3.1"""
    rips = RipsComplex(points=points, max_edge_length=1.0)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    persistence = st.persistence()

    # Extract lifetimes by dimension
    dim_lifetimes = {d: [] for d in range(max_dim + 1)}
    for dim, (birth, death) in persistence:
        if death != float('inf'):
            dim_lifetimes[dim].append(death - birth)

    # 1. Logarithmic transformation
    log_term = sum(np.log1p(l)**2 for lifetimes in dim_lifetimes.values()
                  for l in lifetimes)

    # 2. Enhanced cross-term
    cross_term = 0
    for dim, lifetimes in dim_lifetimes.items():
        if len(lifetimes) > 1:
            lifetimes = np.array(lifetimes)
            cross_term += sum(abs(l1 * l2) for i, l1 in enumerate(lifetimes)
                            for l2 in lifetimes[i+1:])

    # 3. Adaptive scaling
    adaptive_scaling = 1 + np.sqrt(max_dim) * 0.2 + np.exp(0.02 * max_dim)

    # 4. Geometric contribution
    avg_distance = np.mean(pdist(points))

    # 5. Periodic term
    periodic_term = sum(np.sin(np.pi * l / 2) for lifetimes in dim_lifetimes.values()
                       for l in lifetimes)

    k_enhanced = adaptive_scaling * (log_term + cross_term +
                                   np.log1p(len(points) * max_dim) +
                                   periodic_term + avg_distance)

    return k_enhanced, dim_lifetimes

def analyze_hopf_stability(num_trials=100, num_points=100, noise_levels=[0.0, 0.01, 0.02, 0.05]):
    """
    Comprehensive analysis of K-invariant behavior on Hopf fibration
    Tests stability, noise resistance, and structural preservation
    """
    results = {noise: [] for noise in noise_levels}
    persistence_data = {noise: {'s3': [], 's2': []} for noise in noise_levels}

    for noise in noise_levels:
        print(f"\nTesting noise level: {noise}")
        for _ in tqdm(range(num_trials)):
            points_s3, points_s2 = generate_hopf_points(num_points, noise)

            # Compute K-invariants and get persistence data
            k_s3, lifetimes_s3 = compute_enhanced_k_invariant(points_s3, 3)
            k_s2, lifetimes_s2 = compute_enhanced_k_invariant(points_s2, 2)

            ratio = k_s3 / k_s2
            results[noise].append(ratio)

            # Store persistence data
            persistence_data[noise]['s3'].append(lifetimes_s3)
            persistence_data[noise]['s2'].append(lifetimes_s2)

    return results, persistence_data

def plot_results(results, persistence_data):
    """Visualize the analysis results"""
    # Plot ratio distributions
    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    for noise, ratios in results.items():
        plt.hist(ratios, alpha=0.5, label=f'noise={noise}', bins=20)
    plt.xlabel('K(S³)/K(S²) Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of K-invariant Ratios')
    plt.legend()

    # Plot persistence statistics
    plt.subplot(122)
    noise_levels = list(results.keys())
    means = [np.mean(results[n]) for n in noise_levels]
    stds = [np.std(results[n]) for n in noise_levels]

    plt.errorbar(noise_levels, means, yerr=stds, marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Ratio')
    plt.title('Stability of K-invariant Ratio')

    plt.tight_layout()
    plt.show()

def main():
    # Run analysis
    print("Starting Hopf fibration analysis...")
    results, persistence_data = analyze_hopf_stability()

    # Basic statistics
    print("\nResults Summary:")
    for noise, ratios in results.items():
        print(f"\nNoise level: {noise}")
        print(f"Mean ratio: {np.mean(ratios):.4f}")
        print(f"Std ratio: {np.std(ratios):.4f}")
        print(f"Coefficient of variation: {np.std(ratios)/np.mean(ratios):.4f}")

    # Structural tests
    print("\nStructural Consistency Tests:")
    base_ratios = results[0.0]  # No noise case
    mean_base_ratio = np.mean(base_ratios)

    print(f"Base ratio (no noise): {mean_base_ratio:.4f}")
    print(f"Ratio stability (CV): {np.std(base_ratios)/mean_base_ratio:.4f}")

    # Visualize results
    plot_results(results, persistence_data)

if __name__ == "__main__":
    main()
