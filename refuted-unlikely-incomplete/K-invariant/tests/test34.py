import gzip
import networkx as nx
import numpy as np
import random
from collections import defaultdict

# Define the enhanced K_invariant function for simplicity
def k_invariant(points, num_points, dimension, sample_fraction=0.1):
    # Sample a fraction of all possible distances for computational efficiency
    sampled_distances = []
    total_pairs = int(num_points * (num_points - 1) / 2)
    sample_size = int(sample_fraction * total_pairs)

    # Generate random pairs of indices and calculate their distances
    for _ in range(sample_size):
        i, j = random.sample(range(num_points), 2)
        distance = np.linalg.norm(points[i] - points[j])
        sampled_distances.append(distance)

    # Base invariant using a logarithmic transformation of sampled distances
    base_invariant = sum(np.clip(np.log1p(distance) ** 2, 0, 1e6) for distance in sampled_distances)

    # Cross-term to account for interactions using sampled distances
    cross_term = 0
    if len(sampled_distances) > 1:
        for i in range(len(sampled_distances) - 1):
            cross_term += np.clip(abs(sampled_distances[i] * sampled_distances[i + 1]), 0, 1e6)

    # Logarithmic scaling term for minimum complexity bound
    log_term = np.log1p(num_points * dimension)

    # Adaptive scaling factor for dimensional adjustments
    adaptive_scaling = 1 + (dimension ** 0.5) * 0.2 + np.exp(0.02 * dimension)

    # Combining all components for the enhanced K_invariant (clipped to avoid numerical overflow)
    refined_invariant = adaptive_scaling * (base_invariant + cross_term + log_term)
    refined_invariant = np.clip(refined_invariant, 0, 1e12)  # Clip final value to avoid runaway values

    return refined_invariant

# Load the Twitter dataset from the gzipped file
def load_twitter_data(file_path):
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            node1, node2 = line.strip().split()
            G.add_edge(int(node1), int(node2))
    return G

# Extract ego networks and compute K_invariant for each
def analyze_ego_networks(G, sample_size=10, dimension=2, num_points=100):
    ego_k_invariants = {}

    # Randomly select nodes for ego network extraction
    nodes = list(G.nodes)
    sampled_nodes = random.sample(nodes, sample_size)

    for node in sampled_nodes:
        ego_graph = nx.ego_graph(G, node)
        points = nx.to_numpy_array(ego_graph)  # Using adjacency matrix as point cloud representation

        # Normalize points to use as an approximate point cloud
        points = points / np.linalg.norm(points, axis=1, keepdims=True) if points.shape[0] > 1 else points

        # Compute the K_invariant
        k_invariant_value = k_invariant(points, num_points=min(num_points, len(points)), dimension=dimension, sample_fraction=0.1)
        ego_k_invariants[node] = k_invariant_value

    return ego_k_invariants

# Main execution script
def main():
    file_path = "C:\\Users\\cnort\\Desktop\\twitter_combined.txt.gz"

    # Load the social graph
    print("Loading Twitter dataset...")
    G = load_twitter_data(file_path)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Analyze ego networks
    print("Analyzing ego networks...")
    ego_k_invariants = analyze_ego_networks(G, sample_size=20, dimension=3, num_points=50)

    # Output meaningful statistics
    print("\n--- K_invariant Analysis of Ego Networks ---")
    sorted_k_invariants = sorted(ego_k_invariants.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 Users by Ego Network Complexity (K_invariant):")
    for rank, (node, k_value) in enumerate(sorted_k_invariants[:5], start=1):
        print(f"Rank {rank}: User {node} with K_invariant = {k_value:.2f}")

    print(f"\nLowest 5 Users by Ego Network Complexity (K_invariant):")
    for rank, (node, k_value) in enumerate(sorted_k_invariants[-5:], start=1):
        print(f"Rank {rank}: User {node} with K_invariant = {k_value:.2f}")

    avg_k_invariant = np.mean(list(ego_k_invariants.values()))
    print(f"\nAverage K_invariant of sampled ego networks: {avg_k_invariant:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
