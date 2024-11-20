import torch
from tqdm import tqdm
from gudhi import SimplexTree
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
import numpy as np

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the mapping from S^5 to S^3, GPU-accelerated
def sphere_mapping_extended_gpu(theta1, theta2, phi, params):
    a, b, c = [torch.tensor(p, device=device) for p in params]
    theta1, theta2, phi = [torch.tensor(arr, device=device) for arr in [theta1, theta2, phi]]

    x = a * torch.sin(theta1) * torch.cos(phi) * torch.sin(theta2)
    y = b * torch.sin(theta1) * torch.sin(phi) * torch.cos(theta2)
    z = c * torch.cos(theta1)
    w = torch.sin(phi) * torch.cos(theta1) * torch.sin(theta2)
    return torch.stack([x, y, z, w], dim=-1)

# Sparse sampling for manageable dataset size
def sparse_sampling(theta1_values, theta2_values, phi_values, factor=10):
    print(f"Downsampling by factor {factor}...")
    return (
        theta1_values[::factor],
        theta2_values[::factor],
        phi_values[::factor],
    )

# Top-level function for processing a chunk
def process_chunk(args):
    start_idx, chunk_size, points, distance_threshold = args
    end_idx = min(start_idx + chunk_size, len(points))
    tree = cKDTree(points[start_idx:end_idx])
    pairs = tree.query_pairs(distance_threshold)
    return [(start_idx + i, start_idx + j) for i, j in pairs]

# Create a simplicial complex using parallel KDTree processing
def create_simplicial_complex_gpu_parallel(
    theta1_values, theta2_values, phi_values, params, distance_threshold=0.5, chunk_size=10000
):
    simplex_tree = SimplexTree()

    # Generate points on GPU
    print("Generating points on GPU...")
    theta1, theta2, phi = torch.meshgrid(
        torch.tensor(theta1_values, device=device),
        torch.tensor(theta2_values, device=device),
        torch.tensor(phi_values, device=device),
        indexing="ij",
    )
    points = sphere_mapping_extended_gpu(theta1.flatten(), theta2.flatten(), phi.flatten(), params)

    # Transfer points back to CPU
    points = points.cpu().numpy()
    print(f"Total points generated: {len(points)}")

    # Add vertices
    print("Adding vertices...")
    for i in tqdm(range(len(points)), desc="Vertices"):
        simplex_tree.insert([i])

    # Edge addition in parallel
    print("Adding edges using parallel KDTree...")
    args = [
        (i * chunk_size, chunk_size, points, distance_threshold)
        for i in range(len(points) // chunk_size + 1)
    ]
    with Pool(cpu_count()) as pool:
        results = pool.map(process_chunk, args)
        for pairs in results:
            for i, j in pairs:
                simplex_tree.insert([i, j])

    print(f"Total simplices: {simplex_tree.num_simplices()}")
    return simplex_tree, points

# Compute persistent homology for multiple dimensions
def compute_persistent_homology(simplex_tree):
    print("Computing persistent homology...")
    persistence = simplex_tree.persistence()
    h0_barcodes = simplex_tree.persistence_intervals_in_dimension(0)  # H0 features
    h1_barcodes = simplex_tree.persistence_intervals_in_dimension(1)  # H1 features (loops)
    h2_barcodes = simplex_tree.persistence_intervals_in_dimension(2)  # H2 features (voids)
    print("Persistence computation completed.")
    return h0_barcodes, h1_barcodes, h2_barcodes


if __name__ == "__main__":
    # Parameters
    theta1_values = torch.linspace(0, torch.pi, 300).tolist()
    theta2_values = torch.linspace(0, torch.pi, 300).tolist()
    phi_values = torch.linspace(0, 2 * torch.pi, 300).tolist()
    initial_params = [1.0, 1.0, 1.0]

    # Downsample to reduce total points
    theta1_values, theta2_values, phi_values = sparse_sampling(
        theta1_values, theta2_values, phi_values, factor=10
    )

    # Test with perturbed parameters
    param_variations = [
        [1.0, 1.0, 1.0],  # Original
        [0.9, 1.1, 1.0],  # Slightly perturbed
        [1.1, 0.9, 1.0],
        [1.0, 1.0, 0.9],
        [1.0, 1.0, 1.1],
    ]

    for params in param_variations:
        print(f"\nTesting with parameters: {params}")
        simplex_tree, points = create_simplicial_complex_gpu_parallel(
            theta1_values, theta2_values, phi_values, params
        )

        # Compute persistence
        h0_barcodes, h1_barcodes, h2_barcodes = compute_persistent_homology(simplex_tree)

        # Output results for H0, H1, H2
        print("\nPersistent Homology Analysis:")
        print("H0 (Connected Components):")
        for interval in h0_barcodes:
            birth = interval[0]
            death = interval[1] if interval[1] != float('inf') else "Inf"
            print(f"  Birth = {birth:.4f}, Death = {death}")

        print("H1 (Loops):")
        if len(h1_barcodes) > 0:
            for interval in h1_barcodes:
                birth = interval[0]
                death = interval[1] if interval[1] != float('inf') else "Inf"
                print(f"  Birth = {birth:.4f}, Death = {death}")
        else:
            print("  No significant features found in H1 (loops).")

        print("H2 (Voids):")
        if len(h2_barcodes) > 0:
            for interval in h2_barcodes:
                birth = interval[0]
                death = interval[1] if interval[1] != float('inf') else "Inf"
                print(f"  Birth = {birth:.4f}, Death = {death}")
        else:
            print("  No significant features found in H2 (voids).")

        # Statistical Summary
        print(f"\nTotal Points Generated: {len(points)}")
        print(f"Total Simplices in Complex: {simplex_tree.num_simplices()}")
