import gzip
import networkx as nx
import torch
import random
import pandas as pd
from scipy.stats import pearsonr

# Define the enhanced K_invariant function with GPU support using PyTorch
def k_invariant(points, num_points, dimension, sample_fraction=0.5):
    device = points.device  # Ensure computations happen on the same device as the points tensor

    # Convert dimension to a tensor on the correct device
    dimension_tensor = torch.tensor(dimension, dtype=torch.float32, device=device)

    # Sample a larger fraction of all possible distances to leverage the GPU fully
    total_pairs = int(num_points * (num_points - 1) / 2)
    sample_size = int(sample_fraction * total_pairs)

    # Generate random pairs of indices and calculate their distances (now using PyTorch)
    indices = torch.randint(0, num_points, (sample_size, 2), device=device)
    sampled_distances = torch.norm(points[indices[:, 0]] - points[indices[:, 1]], dim=1)

    # Base invariant using a logarithmic transformation of sampled distances
    base_invariant = torch.sum(torch.clamp(torch.log1p(sampled_distances) ** 2, min=0, max=1e6))

    # Cross-term to account for interactions using sampled distances
    cross_term = torch.sum(torch.clamp(torch.abs(sampled_distances[:-1] * sampled_distances[1:]), min=0, max=1e6))

    # Logarithmic scaling term for minimum complexity bound
    log_term = torch.log1p(num_points * dimension_tensor)

    # Adaptive scaling factor for dimensional adjustments
    adaptive_scaling = 1 + (dimension_tensor ** 0.5) * 0.2 + torch.exp(0.02 * dimension_tensor)

    # Combining all components for the enhanced K_invariant (clipped to avoid numerical overflow)
    refined_invariant = adaptive_scaling * (base_invariant + cross_term + log_term)
    refined_invariant = torch.clamp(refined_invariant, min=0, max=1e12)  # Clip final value to avoid runaway values

    return refined_invariant.item()  # Convert back to Python scalar

# Load the Twitter dataset from the gzipped file
def load_twitter_data(file_path):
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            node1, node2 = line.strip().split()
            G.add_edge(int(node1), int(node2))
    return G

# Extract ego networks and compute K_invariant for each
def analyze_ego_networks(G, sample_size=10, dimensions=[2, 3, 4], num_points=100, stepwise_sample_sizes=[10, 20, 50]):
    results = []

    # Randomly select nodes for ego network extraction
    nodes = list(G.nodes)
    sampled_nodes = random.sample(nodes, sample_size)

    # Convert NetworkX graph to a PyTorch-based adjacency matrix for GPU operations
    adjacency_matrix = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.tensor(adjacency_matrix.row, dtype=torch.long, device='cuda')
    col = torch.tensor(adjacency_matrix.col, dtype=torch.long, device='cuda')
    data = torch.tensor(adjacency_matrix.data, dtype=torch.float32, device='cuda')
    gpu_matrix = torch.sparse_coo_tensor(torch.vstack((row, col)), data, size=adjacency_matrix.shape)

    # Calculate degree centrality entirely on the GPU
    degrees = torch.sparse.sum(gpu_matrix, dim=1).to_dense()
    degree_centrality = degrees / (len(nodes) - 1)

    # Calculate clustering coefficients directly on GPU
    adjacency_csr = gpu_matrix.to_dense()  # For clustering coefficients, we convert to a dense matrix
    clustering_coefficients = torch.zeros(len(nodes), dtype=torch.float32, device='cuda')

    for node in range(len(nodes)):
        neighbors = torch.nonzero(adjacency_csr[node]).squeeze()

        # Ensure `neighbors` is always treated as a 1D tensor
        if neighbors.ndimension() == 0:
            neighbors = neighbors.unsqueeze(0)

        if neighbors.shape[0] < 2:
            clustering_coefficients[node] = 0.0
        else:
            subgraph = adjacency_csr[neighbors][:, neighbors]
            possible_links = neighbors.shape[0] * (neighbors.shape[0] - 1)
            actual_links = subgraph.sum() / 2  # Each edge is counted twice in an undirected graph
            clustering_coefficients[node] = actual_links / possible_links

    for sample_count in stepwise_sample_sizes:
        print(f"Analyzing with sample size: {sample_count}")
        for dimension in dimensions:
            for node in sampled_nodes[:sample_count]:
                # Extract ego graph and move adjacency matrix to GPU
                ego_graph = nx.ego_graph(G, node)
                points = torch.tensor(nx.to_numpy_array(ego_graph), dtype=torch.float32, device='cuda')

                # Normalize points to use as an approximate point cloud
                if points.shape[0] > 1:
                    points = points / torch.norm(points, dim=1, keepdim=True)

                # Compute the K_invariant
                k_invariant_value = k_invariant(points, num_points=min(num_points, len(points)), dimension=dimension, sample_fraction=0.5)

                # Extract centrality measures for the ego node (directly from GPU array)
                node_index = nodes.index(node)
                betweenness_centrality = nx.betweenness_centrality(G)[node]  # Still CPU-based, but for sampling
                degree_centrality_value = degree_centrality[node_index].item()
                clustering_coefficient_value = clustering_coefficients[node_index].item()

                # Store results for analysis
                results.append({
                    "node": node,
                    "sample_size": sample_count,
                    "dimension": dimension,
                    "k_invariant": k_invariant_value,
                    "degree_centrality": degree_centrality_value,
                    "betweenness_centrality": betweenness_centrality,
                    "clustering_coefficient": clustering_coefficient_value
                })

    # Create a DataFrame for analysis
    df = pd.DataFrame(results)
    return df

# Main execution script
def main():
    file_path = "C:\\Users\\cnort\\Desktop\\twitter_combined.txt.gz"

    # Load the social graph
    print("Loading Twitter dataset...")
    G = load_twitter_data(file_path)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Analyze ego networks with stepwise sample sizes and different dimensions
    print("Analyzing ego networks with varying parameters...")
    analysis_df = analyze_ego_networks(G, sample_size=30, dimensions=[2, 3, 5], num_points=50, stepwise_sample_sizes=[10, 20])

    # Output statistical analysis of the results
    print("\n--- Statistical Analysis of K_invariant with Social Metrics ---")

    for dimension in analysis_df["dimension"].unique():
        df_dimension = analysis_df[analysis_df["dimension"] == dimension]
        print(f"\nDimension: {dimension}")
        print(f"Average K_invariant: {df_dimension['k_invariant'].mean():.2f}")
        print(f"Standard Deviation of K_invariant: {df_dimension['k_invariant'].std():.2f}")

        # Correlation analysis
        for metric in ["degree_centrality", "betweenness_centrality", "clustering_coefficient"]:
            corr, p_value = pearsonr(df_dimension["k_invariant"], df_dimension[metric])
            print(f"Correlation between K_invariant and {metric}: r = {corr:.2f}, p-value = {p_value:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
