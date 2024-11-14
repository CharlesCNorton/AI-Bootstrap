import numpy as np
import networkx as nx
from numpy.linalg import eigh
import pandas as pd
import random

# Utility function to determine if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    for n in range(2, int(num ** 0.5) + 1):
        if num % n == 0:
            return False
    return True

# Step 1: Initialize the 2D Cellular Automaton State
N_2D = 50  # Grid size (NxN)
iterations = 20  # Number of iterations for evolution

# Create initial state based on primality
initial_state_2D = np.zeros((N_2D, N_2D))
for i in range(N_2D):
    for j in range(N_2D):
        number = (i + 1) * (j + 1)  # Product of row and column indices
        initial_state_2D[i, j] = 1 if is_prime(number) else 0

# Step 2: Define Updated Probabilistic Update Rules for the 2D Cellular Automaton
def update_state_2D_probabilistic(current_state):
    rows, cols = current_state.shape
    next_state = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            number = (i + 1) * (j + 1)

            # Define the Moore neighborhood (8 neighbors)
            alive_neighbors = 0
            for ni in [-1, 0, 1]:
                for nj in [-1, 0, 1]:
                    if ni == 0 and nj == 0:
                        continue  # Skip the cell itself
                    neighbor_row, neighbor_col = i + ni, j + nj
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        alive_neighbors += current_state[neighbor_row, neighbor_col]

            # Apply probabilistic rules
            if current_state[i, j] == 1:
                # If the cell is already alive, it remains alive
                next_state[i, j] = 1
            else:
                # Calculate probability of becoming alive based on number of alive neighbors
                prob = min(1, alive_neighbors / 8)  # Probability based on the fraction of alive neighbors
                if random.random() < prob:
                    next_state[i, j] = 1
                else:
                    next_state[i, j] = 0

    return next_state

# Step 3: Simulate the Evolution of the 2D Cellular Automaton with Probabilistic Updates
state_history_2D_prob = [initial_state_2D]
current_state_2D = initial_state_2D
for _ in range(iterations):
    current_state_2D = update_state_2D_probabilistic(current_state_2D)
    state_history_2D_prob.append(current_state_2D)

# Step 4: Convert 2D CA State to Graph Representation
def ca_state_to_graph(ca_state):
    rows, cols = ca_state.shape
    G = nx.Graph()

    # Add nodes for each cell
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j), state=ca_state[i, j])

    # Add edges for neighboring cells (4-neighborhood)
    for i in range(rows):
        for j in range(cols):
            current_node = (i, j)
            neighbors = [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
            ]
            for neighbor in neighbors:
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    G.add_edge(current_node, neighbor)

    return G

# Step 5: Spectral Analysis of the Graph
def get_graph_spectral_properties(graph):
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()

    # Compute the eigenvalues using a dense solver
    eigenvalues = eigh(laplacian_matrix)[0]

    # Get non-zero eigenvalues
    non_zero_eigenvalues = [ev for ev in eigenvalues if ev > 1e-6]

    # Return the two smallest non-zero eigenvalues
    return non_zero_eigenvalues[:2] if len(non_zero_eigenvalues) >= 2 else non_zero_eigenvalues + [0] * (2 - len(non_zero_eigenvalues))

# Step 6: Perform Spectral Analysis and Collect Statistics
spectral_properties_over_time = []
local_clustering_over_time = []

for state in state_history_2D_prob:
    graph = ca_state_to_graph(state)

    # Spectral properties
    spectral_properties = get_graph_spectral_properties(graph)
    spectral_properties_over_time.append(spectral_properties)

    # Local Clustering Coefficient (average)
    clustering_coeffs = nx.clustering(graph)
    avg_clustering_coeff = sum(clustering_coeffs.values()) / len(clustering_coeffs)
    local_clustering_over_time.append(avg_clustering_coeff)

# Step 7: Convert the Collected Data into a DataFrame for Comprehensive Analysis
spectral_df_prob = pd.DataFrame(spectral_properties_over_time, columns=['Eigenvalue_1', 'Eigenvalue_2'])
spectral_df_prob['Avg_Clustering_Coefficient'] = local_clustering_over_time

# Step 8: Calculate Summary Statistics
summary_statistics_prob = spectral_df_prob.describe()

# Output Descriptive Statistics for the Updated CA Evolution
print("\n===== Comprehensive Summary Statistics for Probabilistic 2D Cellular Automaton Evolution =====")
print(summary_statistics_prob)

# Output Correlation Matrix for Graph Metrics Over Time
correlation_matrix_prob = spectral_df_prob.corr()
print("\n===== Correlation Matrix for Graph Metrics Over Time (Probabilistic CA) =====")
print(correlation_matrix_prob)

# Output Key Spectral Properties and Average Clustering Coefficient for Selected Iterations
print("\n===== Selected Iteration Data =====")
selected_iterations = [0, 5, 10, 15, 20]  # Specific iterations for concise output
for iteration in selected_iterations:
    eigen1, eigen2 = spectral_properties_over_time[iteration]
    avg_clustering = local_clustering_over_time[iteration]
    print(f"Iteration {iteration}: Eigenvalue_1 = {eigen1:.6f}, Eigenvalue_2 = {eigen2:.6f}, Avg_Clustering_Coefficient = {avg_clustering:.6f}")

# Print final row of collected data to observe ending state
print("\n===== Final Iteration Data =====")
print(spectral_df_prob.tail(1))
