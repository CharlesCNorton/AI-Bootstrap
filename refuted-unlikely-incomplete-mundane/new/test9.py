import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# Constants from the theorem
R0 = 1.039  # Empirical constant for radial scaling
sqrt_3 = np.sqrt(3)

# Helper function to compute predicted radial distances
def predicted_radial(dim):
    return R0 * (1 / sqrt_3) ** dim

# Step 1: Analyze Radial Distance Scaling
def analyze_radial_distances(diagrams):
    results = []
    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0:
            continue

        # Extract birth, death, and compute radial distances
        birth, death = diagram[:, 0], diagram[:, 1]
        finite_mask = np.isfinite(birth) & np.isfinite(death)
        birth, death = birth[finite_mask], death[finite_mask]
        radial_distances = np.sqrt(birth**2 + death**2)

        # Compare to predicted radial distance
        pred = predicted_radial(dim)
        errors = np.abs(radial_distances - pred)

        # Save results
        results.append({
            "dimension": dim,
            "mean_radial_distance": np.mean(radial_distances),
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "radial_distances": radial_distances.tolist(),
            "errors": errors.tolist(),
        })

    return results

# Step 2: Analyze Dimensional Scaling
def analyze_dimensional_scaling(diagrams):
    scaling_results = []
    for dim in range(len(diagrams) - 1):
        current = diagrams[dim]
        next_dim = diagrams[dim + 1]
        if len(current) == 0 or len(next_dim) == 0:
            continue

        # Compute radial distances
        current_radial = np.sqrt(current[:, 0]**2 + current[:, 1]**2)
        next_radial = np.sqrt(next_dim[:, 0]**2 + next_dim[:, 1]**2)

        # Handle mismatched lengths by truncating to the shorter array
        min_length = min(len(current_radial), len(next_radial))
        current_radial = current_radial[:min_length]
        next_radial = next_radial[:min_length]

        # Compute scaling ratios
        ratios = next_radial / current_radial

        # Save results
        scaling_results.append({
            "from_dim": dim,
            "to_dim": dim + 1,
            "mean_ratio": np.mean(ratios),
            "std_ratio": np.std(ratios),
            "ratios": ratios.tolist(),
        })

    return scaling_results

# Step 3: Analyze Persistence Lengths
def analyze_persistence_lengths(diagrams):
    results = []
    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0:
            continue

        # Compute persistence lengths
        persistence_lengths = diagram[:, 1] - diagram[:, 0]
        finite_mask = np.isfinite(persistence_lengths)
        persistence_lengths = persistence_lengths[finite_mask]

        # Save results
        results.append({
            "dimension": dim,
            "mean_persistence_length": np.mean(persistence_lengths),
            "max_persistence_length": np.max(persistence_lengths),
            "persistence_lengths": persistence_lengths.tolist(),
        })

    return results

# Step 4: Main Analysis Function
def analyze_theorem(diagrams):
    # Perform analyses
    radial_results = analyze_radial_distances(diagrams)
    scaling_results = analyze_dimensional_scaling(diagrams)
    persistence_results = analyze_persistence_lengths(diagrams)

    # Print summary
    print("\n=== Radial Distance Analysis ===")
    for result in radial_results:
        print(f"Dimension {result['dimension']}:")
        print(f"  Mean Radial Distance: {result['mean_radial_distance']:.4f}")
        print(f"  Mean Error: {result['mean_error']:.4f}")
        print(f"  Max Error: {result['max_error']:.4f}")

    print("\n=== Dimensional Scaling Analysis ===")
    for result in scaling_results:
        print(f"From Dimension {result['from_dim']} to {result['to_dim']}:")
        print(f"  Mean Ratio: {result['mean_ratio']:.4f}")
        print(f"  Std Dev Ratio: {result['std_ratio']:.4f}")

    print("\n=== Persistence Length Analysis ===")
    for result in persistence_results:
        print(f"Dimension {result['dimension']}:")
        print(f"  Mean Persistence Length: {result['mean_persistence_length']:.4f}")
        print(f"  Max Persistence Length: {result['max_persistence_length']:.4f}")

# Generate a noisy sphere and analyze the theorem
if __name__ == "__main__":
    # Generate noisy sphere
    np.random.seed(42)
    phi = np.random.uniform(0, 2 * np.pi, 300)
    theta = np.random.uniform(0, np.pi, 300)
    x = np.sin(theta) * np.cos(phi) + 0.1 * np.random.randn(300)
    y = np.sin(theta) * np.sin(phi) + 0.1 * np.random.randn(300)
    z = np.cos(theta) + 0.1 * np.random.randn(300)
    point_cloud = np.column_stack((x, y, z))

    # Compute persistence diagram
    diagrams = ripser(point_cloud, maxdim=2)['dgms']

    # Analyze theorem
    analyze_theorem(diagrams)
