import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import pandas as pd

# Helper functions for numerical methods
def graph_eigenvalue_method(G, c, t_max, dt):
    start_time = time.perf_counter()
    L = nx.laplacian_matrix(G).asfptype()
    n_nodes = L.shape[0]

    # Compute eigenvalues and eigenvectors
    k = min(n_nodes - 2, 200)  # Handle more modes for complex behavior
    eigenvalues, eigenvectors = spla.eigsh(L, k=k, which='SM')
    eigenvalues[eigenvalues < 0] = 0.0
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    # Initial conditions: impulse at center node
    center_node = n_nodes // 2
    u0 = np.zeros(n_nodes)
    u0[center_node] = 1.0

    # Project initial condition onto eigenvectors
    a0 = eigenvectors.T @ u0
    t_values = np.arange(0, t_max + dt, dt)
    a_values = np.zeros((len(eigenvalues), len(t_values)))

    # Solve ODEs for each eigenmode
    for i, (lambda_i, a0_i) in enumerate(zip(eigenvalues, a0)):
        omega_i = c * np.sqrt(lambda_i)
        a_values[i, :] = a0_i * np.cos(omega_i * t_values) if omega_i != 0 else a0_i

    # Reconstruct the wave at each node over time
    u_values = eigenvectors @ a_values
    computation_time = time.perf_counter() - start_time
    return u_values, t_values, computation_time

def finite_difference_method(G, c, t_max, dt):
    start_time = time.perf_counter()
    n_nodes = G.number_of_nodes()
    dx = 1.0
    t_values = np.arange(0, t_max + dt, dt)
    u_values = np.zeros((n_nodes, len(t_values)))

    # Initial conditions: impulse at center node
    center_node = n_nodes // 2
    u_values[center_node, 0] = 1.0

    # CFL condition
    s = (c * dt / dx) ** 2
    if s > 1:
        raise ValueError("CFL condition not met.")

    # Time-stepping loop
    for n in range(1, len(t_values) - 1):
        u_values[1:-1, n + 1] = (2 * (1 - s) * u_values[1:-1, n] - u_values[1:-1, n - 1] +
                                 s * (u_values[2:, n] + u_values[:-2, n]))
        u_values[0, n + 1] = 0.0
        u_values[-1, n + 1] = 0.0

    computation_time = time.perf_counter() - start_time
    return u_values, t_values, computation_time

def finite_element_method(G, c, t_max, dt):
    start_time = time.perf_counter()
    n_nodes = G.number_of_nodes()
    L = nx.laplacian_matrix(G).asfptype()
    M = sp.identity(n_nodes, format='csr')
    t_values = np.arange(0, t_max + dt, dt)
    u_values = np.zeros((n_nodes, len(t_values)))

    # Initial conditions: impulse at center node
    center_node = n_nodes // 2
    u_values[center_node, 0] = 1.0

    # Newmark-beta method parameters
    beta = 0.25
    gamma = 0.5
    u_prev = u_values[:, 0]
    v_prev = np.zeros(n_nodes)
    a_prev = sp.linalg.spsolve(M, -L @ u_prev)

    for n in range(1, len(t_values)):
        delta_u = dt * v_prev + (dt ** 2 * (0.5 - beta) * a_prev)
        u_curr = u_prev + delta_u
        LHS = M + beta * dt ** 2 * L
        RHS = -L @ u_curr
        a_curr = sp.linalg.spsolve(LHS, RHS)
        v_curr = v_prev + dt * ((1 - gamma) * a_prev + gamma * a_curr)
        u_values[:, n] = u_curr

        u_prev = u_curr
        v_prev = v_curr
        a_prev = a_curr

    computation_time = time.perf_counter() - start_time
    return u_values, t_values, computation_time

def spectral_method(G, c, t_max, dt):
    start_time = time.perf_counter()
    n_nodes = G.number_of_nodes()
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues[eigenvalues < 0] = 0.0
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    # Initial conditions: impulse at center node
    center_node = n_nodes // 2
    u0 = np.zeros(n_nodes)
    u0[center_node] = 1.0

    a0 = eigenvectors.T @ u0
    t_values = np.arange(0, t_max + dt, dt)
    a_values = np.zeros((len(eigenvalues), len(t_values)))

    # Solve ODEs for each eigenmode
    for i, (lambda_i, a0_i) in enumerate(zip(eigenvalues, a0)):
        omega_i = c * np.sqrt(lambda_i)
        a_values[i, :] = a0_i * np.cos(omega_i * t_values) if omega_i != 0 else a0_i

    u_values = eigenvectors @ a_values
    computation_time = time.perf_counter() - start_time
    return u_values, t_values, computation_time

# Generate large, complex artificial graphs
def generate_large_test_graphs(n_nodes):
    return {
        'Large Small-World Graph': nx.watts_strogatz_graph(n_nodes, k=10, p=0.3, seed=42),
        'Large Scale-Free Graph': nx.barabasi_albert_graph(n_nodes, m=5, seed=42)
    }

# Function to simulate dynamic changes in the network
def modify_graph_dynamically(G, num_changes=10):
    for _ in range(num_changes):
        if G.number_of_edges() > 0:
            edge_to_remove = list(G.edges())[np.random.randint(0, G.number_of_edges())]
            G.remove_edge(*edge_to_remove)
            node1, node2 = np.random.choice(list(G.nodes()), 2, replace=False)
            G.add_edge(node1, node2)
    return G

# Run comparative analysis on large dynamic networks
def run_faithful_comparative_analysis(n_nodes, c, t_max, dt, num_changes=10):
    graphs = generate_large_test_graphs(n_nodes)
    methods = {
        'Graph Eigenvalue Method': graph_eigenvalue_method,
        'Finite Difference Method': finite_difference_method,
        'Finite Element Method': finite_element_method,
        'Spectral Method': spectral_method
    }

    results = []
    for graph_name, G in graphs.items():
        print(f"\nInitial analysis on {graph_name}...")
        for method_name, method in methods.items():
            print(f"Running {method_name} on {graph_name}...")
            try:
                u_values, t_values, computation_time = method(G, c, t_max, dt)
                results.append({
                    'Graph Type': graph_name,
                    'Method': method_name,
                    'Initial Computation Time (s)': computation_time
                })
            except Exception as e:
                print(f"Error running {method_name} on {graph_name}: {e}")
                results.append({
                    'Graph Type': graph_name,
                    'Method': method_name,
                    'Initial Computation Time (s)': float('inf')
                })

        # Modify the graph to simulate dynamic changes
        G = modify_graph_dynamically(G, num_changes)
        print(f"\nDynamic analysis on modified {graph_name}...")

        for method_name, method in methods.items():
            print(f"Re-running {method_name} on modified {graph_name}...")
            try:
                u_values, t_values, computation_time = method(G, c, t_max, dt)
                results.append({
                    'Graph Type': graph_name + ' (Modified)',
                    'Method': method_name,
                    'Post-Modification Computation Time (s)': computation_time
                })
            except Exception as e:
                print(f"Error running {method_name} on modified {graph_name}: {e}")
                results.append({
                    'Graph Type': graph_name + ' (Modified)',
                    'Method': method_name,
                    'Post-Modification Computation Time (s)': float('inf')
                })

    results_df = pd.DataFrame(results)
    print("\n--- Faithful Comparative Analysis Results ---")
    print(results_df)

# Execute the faithful comparative analysis with large, dynamic graphs
run_faithful_comparative_analysis(n_nodes=5000, c=1.0, t_max=1.0, dt=0.01)
