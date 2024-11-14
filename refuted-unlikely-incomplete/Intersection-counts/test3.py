import networkx as nx
import numpy as np
import pandas as pd

# Constants for propagation factor calculation based on the theorem
a1 = 0.00036  # Weight for average degree (k_avg), small to reflect limited influence
a2 = 0.00929  # Weight for clustering coefficient (dominant factor)
a3 = 0.00075  # Weight for mean shortest path length, secondary influence

def calculate_alpha(k_avg, C_clustering, L_path):
    # The propagation factor alpha gives primary importance to clustering coefficient
    return a1 * k_avg + a2 * C_clustering + a3 * L_path

def predict_failure_likelihood(alpha):
    # Predict cascading failure likelihood, scaled to 1 at max
    return min(1, alpha * 10)

# Test parameters
num_networks = 1000  # Number of interdependent networks
network_size_range = (20, 500)  # Network sizes between 20 and 500 nodes
edge_change_rate = 0.4  # 40% of edges change per time step
time_steps = 10  # Number of time steps to simulate network changes

print("Initializing networks with random types and sizes...")

# Initialize networks and track type counts for statistical analysis
network_types = ["ER", "WS", "BA", "Grid"]
np.random.seed(42)
networks = []
network_type_count = {ntype: 0 for ntype in network_types}

for _ in range(num_networks):
    n_nodes = np.random.randint(*network_size_range)
    network_type = np.random.choice(network_types)

    if network_type == "ER":
        G = nx.erdos_renyi_graph(n_nodes, 0.05)
    elif network_type == "WS":
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.2)
    elif network_type == "BA":
        G = nx.barabasi_albert_graph(n_nodes, 3)
    elif network_type == "Grid":
        G = nx.grid_2d_graph(int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)))

    networks.append(G)
    network_type_count[network_type] += 1

print("Network initialization complete. Type distribution:")
print(network_type_count)

# Function for simulating severe changes in network structure
def severe_dynamic_change(G, edge_change_rate=0.4):
    nodes = list(G.nodes())
    edges = list(G.edges())

    # Remove edges randomly based on severe change rate
    for edge in edges:
        if np.random.rand() < edge_change_rate:
            if G.has_edge(*edge):
                G.remove_edge(*edge)

    # Randomly add new edges between nodes to simulate instability
    for _ in range(int(edge_change_rate * len(edges))):
        node1, node2 = np.random.choice(range(len(nodes)), 2, replace=False)
        if not G.has_edge(nodes[node1], nodes[node2]):
            G.add_edge(nodes[node1], nodes[node2])

    return G

# Lists to collect results for statistical analysis
alphas = []
failure_likelihoods = []
clustering_coeffs = []
avg_degrees = []
path_lengths = []

print("Running the simulation over", time_steps, "time steps...")

for t in range(time_steps):
    print(f"Time Step {t + 1}/{time_steps}")
    updated_networks = [severe_dynamic_change(G.copy()) if np.random.rand() < 0.5 else G for G in networks]

    for i, G in enumerate(updated_networks):
        # Calculate clustering coefficient, average degree, and mean path length
        C_clustering = nx.average_clustering(G)
        k_avg = np.mean([deg for _, deg in G.degree()])
        L_path = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')

        # Calculate alpha according to the theorem, with clustering as dominant factor
        alpha = calculate_alpha(k_avg, C_clustering, L_path)
        failure_likelihood = predict_failure_likelihood(alpha)

        # Collect data for statistical summary
        alphas.append(alpha)
        failure_likelihoods.append(failure_likelihood)
        clustering_coeffs.append(C_clustering)
        avg_degrees.append(k_avg)
        path_lengths.append(L_path if L_path != float('inf') else np.nan)  # Exclude infinite path lengths

    print(f"  Processed networks for time step {t + 1}")

# Remove infinite values from alphas before calculating statistics
finite_alphas = [x for x in alphas if x != float('inf')]
finite_failure_likelihoods = [x for x in failure_likelihoods if x != float('inf')]
finite_path_lengths = [x for x in path_lengths if not np.isnan(x)]

# Print statistical summary
print("\n===== Statistical Summary of Large-Scale Cascading Failure Test =====")
print("Network Types Distribution:", network_type_count)
print("\nPropagation Factor (Alpha) Statistics:")
print(f"  Min: {np.min(finite_alphas):.6f}")
print(f"  Max: {np.max(finite_alphas):.6f}")
print(f"  Mean: {np.mean(finite_alphas):.6f}")
print(f"  Std Dev: {np.std(finite_alphas):.6f}")

print("\nFailure Likelihood Statistics:")
print(f"  Min: {np.min(finite_failure_likelihoods):.6f}")
print(f"  Max: {np.max(finite_failure_likelihoods):.6f}")
print(f"  Mean: {np.mean(finite_failure_likelihoods):.6f}")
print(f"  Std Dev: {np.std(finite_failure_likelihoods):.6f}")

print("\nAverage Network Properties Across All Time Steps:")
print(f"  Clustering Coefficient (Mean): {np.nanmean(clustering_coeffs):.6f}")
print(f"  Average Degree (Mean): {np.nanmean(avg_degrees):.6f}")
print(f"  Mean Path Length (Mean, excluding inf): {np.mean(finite_path_lengths):.6f}")

print("=====================================================================")
