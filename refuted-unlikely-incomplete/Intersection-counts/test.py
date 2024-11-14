import networkx as nx
import numpy as np
import concurrent.futures
import time

# Function to create and process a single network
def create_and_simulate_network(network_type, num_nodes=1_000_000):
    print(f"Starting creation of {network_type} network...")

    start_time = time.time()

    if network_type == "Barabási-Albert (Scale-Free)":
        graph = nx.barabasi_albert_graph(num_nodes, 3)
    elif network_type == "Watts-Strogatz (Small-World)":
        graph = nx.watts_strogatz_graph(num_nodes, 6, 0.1)
    elif network_type == "Erdős-Rényi (Random)":
        graph = nx.erdos_renyi_graph(num_nodes, 0.001)
    elif network_type == "Hierarchical Tree":
        graph = nx.balanced_tree(3, 10)
    elif network_type == "Complete Graph":
        graph = nx.complete_graph(min(num_nodes, 5000))  # Limit size for complete graphs
    elif network_type == "Grid Network (2D Grid)":
        side_length = int(np.sqrt(num_nodes))
        graph = nx.grid_2d_graph(side_length, side_length)
    elif network_type == "Random Geometric":
        graph = nx.random_geometric_graph(num_nodes, 0.02)
    elif network_type == "Power Law Cluster":
        graph = nx.powerlaw_cluster_graph(num_nodes, 3, 0.1)
    elif network_type == "Scale-Free (Power Law Degree)":
        graph = nx.scale_free_graph(num_nodes)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    end_time = time.time()
    print(f"Completed creation of {network_type} in {end_time - start_time:.2f} seconds.")

    # Simulate cascading failures immediately after creating the network
    print(f"Starting cascading failure simulation for {network_type}...")
    connected_components_data = simulate_failure(graph)
    statistics = calculate_statistics(connected_components_data)

    # Output summary statistics for the completed network
    print(f"\nSummary for {network_type}:")
    for key, value in statistics.items():
        if isinstance(value, list):
            print(f"{key}: First few values: {value[:5]} ... Last few values: {value[-5:]}")
    print(f"Completed simulation for network: {network_type}\n")

# Function to simulate cascading failures on a given network
def simulate_failure(graph, initial_failure_fraction=0.05):
    num_nodes = len(graph)
    initial_failures = int(initial_failure_fraction * num_nodes)

    nodes_list = list(graph.nodes())
    failed_nodes = np.random.choice(nodes_list, initial_failures, replace=False)

    sizes_of_components = []
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(failed_nodes)

    sizes_of_components.append([len(c) for c in nx.connected_components(graph_copy)])

    iteration = 0
    while True:
        iteration += 1
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Number of nodes left = {graph_copy.number_of_nodes()}")

        to_fail_next = set()
        for node in failed_nodes:
            if node in graph_copy:
                neighbors = list(graph_copy.neighbors(node))
                to_fail_next.update(neighbors)

        if not to_fail_next:
            break

        failed_nodes = list(to_fail_next)
        graph_copy.remove_nodes_from(failed_nodes)
        sizes_of_components.append([len(c) for c in nx.connected_components(graph_copy)])

    return sizes_of_components

# Function to calculate statistical properties from the connected components data
def calculate_statistics(connected_components_data):
    print("Calculating statistics...")
    sizes_over_iterations = [np.mean(iteration) for iteration in connected_components_data]
    max_sizes_over_iterations = [np.max(iteration) for iteration in connected_components_data]
    min_sizes_over_iterations = [np.min(iteration) for iteration in connected_components_data]

    stats = {
        "mean_sizes_per_iteration": sizes_over_iterations,
        "max_sizes_per_iteration": max_sizes_over_iterations,
        "min_sizes_per_iteration": min_sizes_over_iterations
    }

    print("Statistics calculation completed.")
    return stats

# Main function to run the entire simulation
def main():
    network_types = [
        "Barabási-Albert (Scale-Free)",
        "Watts-Strogatz (Small-World)",
        "Erdős-Rényi (Random)",
        "Hierarchical Tree",
        "Complete Graph",
        "Grid Network (2D Grid)",
        "Random Geometric",
        "Power Law Cluster",
        "Scale-Free (Power Law Degree)"
    ]

    # Set max_workers to the number of network types for concurrent processing
    max_workers = len(network_types)
    print(f"Using max_workers = {max_workers} for parallel execution.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each network creation and simulation task to the executor
        futures = {executor.submit(create_and_simulate_network, network_type): network_type for network_type in network_types}

        for future in concurrent.futures.as_completed(futures):
            network_type = futures[future]
            try:
                future.result()  # Get the result of the future to ensure any exceptions are raised
            except Exception as exc:
                print(f"Simulation generated an exception for {network_type}: {exc}")

if __name__ == "__main__":
    main()
