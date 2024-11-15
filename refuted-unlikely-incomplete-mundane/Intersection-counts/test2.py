import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute network metrics: avg degree, clustering coefficient, mean shortest path length
def compute_network_metrics(G):
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    clustering_coefficient = nx.average_clustering(G)
    try:
        shortest_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # Handle disconnected graphs by approximating with largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        shortest_path_length = nx.average_shortest_path_length(G_largest)

    return avg_degree, clustering_coefficient, shortest_path_length

# Function to compute network metrics for multiple instances and average them
def compute_average_metrics(network_func, num_trials=5, *args, **kwargs):
    avg_deg_list, cluster_coef_list, mean_path_len_list = [], [], []

    for _ in range(num_trials):
        G = network_func(*args, **kwargs)
        avg_deg, cluster_coef, mean_path_len = compute_network_metrics(G)
        avg_deg_list.append(avg_deg)
        cluster_coef_list.append(cluster_coef)
        mean_path_len_list.append(mean_path_len)

    avg_deg = np.mean(avg_deg_list)
    cluster_coef = np.mean(cluster_coef_list)
    mean_path_len = np.mean(mean_path_len_list)

    return avg_deg, cluster_coef, mean_path_len

# Coefficients provided for propagation factor computation
a1, a2, a3 = 0.00036, 0.00929, 0.00075

# Step 1: Define network types and parameters
network_types_extended = {
    'Erdos_Renyi': lambda: nx.erdos_renyi_graph(5000, 0.01),
    'Scale_Free': lambda: nx.Graph(nx.scale_free_graph(5000)),
    'Small_World': lambda: nx.watts_strogatz_graph(5000, 10, 0.1),
    'Barabasi_Albert': lambda: nx.barabasi_albert_graph(5000, 5),
    'Random_Geometric': lambda: nx.random_geometric_graph(5000, 0.05)
}

# Step 2: Calculate metrics for each extended network type over multiple trials
network_metrics_extended = {}
for net_name, network_func in network_types_extended.items():
    avg_deg, cluster_coef, mean_path_len = compute_average_metrics(network_func, num_trials=3)
    network_metrics_extended[net_name] = {
        'Average Degree': avg_deg,
        'Clustering Coefficient': cluster_coef,
        'Mean Shortest Path Length': mean_path_len
    }

# Step 3: Store results in a DataFrame for further analysis
metrics_df_extended = pd.DataFrame(network_metrics_extended).T

# Step 4: Compute propagation factor alpha using given coefficients
metrics_df_extended['Propagation Factor (Alpha)'] = (
    a1 * metrics_df_extended['Average Degree'] +
    a2 * metrics_df_extended['Clustering Coefficient'] +
    a3 * metrics_df_extended['Mean Shortest Path Length']
)

# Step 5: Display results
print("Extended Network Metrics and Propagation Factor for Different Network Topologies:")
print(metrics_df_extended)

# Optional: Plotting the results for better visualization
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df_extended['Propagation Factor (Alpha)'].plot(kind='bar', ax=ax, color='skyblue')
ax.set_ylabel('Propagation Factor (Alpha)')
ax.set_title('Propagation Factor (Alpha) across Different Network Topologies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
