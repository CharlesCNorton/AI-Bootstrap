import tarfile
import os
import networkx as nx
import gzip
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigsh as cpu_eigsh
from scipy.integrate import solve_ivp
import time

# Paths to the data files as provided by the user
tar_file_path = "D:\\Twitter\\twitter.tar.gz"
combined_file_path = "D:\\Twitter\\twitter_combined.txt.gz"

# Step 1: Extract the Twitter data from the tar.gz file
print("\nStep 1: Extracting the Twitter data...")
extracted_dir = "D:\\Twitter\\extracted"
with tarfile.open(tar_file_path, "r:gz") as tar:
    tar.extractall(path=extracted_dir)
print("Data extraction complete.")

# Step 2: Initialize an empty directed graph to store all combined data
print("\nStep 2: Initializing an empty directed graph...")
G_combined = nx.DiGraph()

# Step 3: Parse the extracted ego network data to construct the graph
print("\nStep 3: Parsing extracted ego network data...")
for root, _, files in os.walk(extracted_dir):
    for file in files:
        if file.endswith(".edges"):
            with open(os.path.join(root, file), 'r') as f:
                for line in f:
                    node_a, node_b = line.strip().split()
                    G_combined.add_edge(node_a, node_b)
print("Parsing complete for ego network data.")

# Step 4: Additionally, parse the combined edges from twitter_combined.txt.gz
print("\nStep 4: Parsing combined edges from twitter_combined.txt.gz...")
with gzip.open(combined_file_path, 'rt') as f:
    for line in f:
        node_a, node_b = line.strip().split()
        G_combined.add_edge(node_a, node_b)
print("Parsing complete for combined edges.")

# Step 5: Display initial information about the combined graph
print("\n--- Combined Twitter Network Graph Information Before Modifications ---")
num_nodes = G_combined.number_of_nodes()
num_edges = G_combined.number_of_edges()
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")
print(f"Is Directed: {G_combined.is_directed()}")

# Step 6: Convert the graph to undirected and compute the Laplacian matrix using CuPy
print("\nStep 6: Preparing for wave propagation simulation...")
G_undirected = G_combined.to_undirected()

# Start timing for Laplacian calculation
start_time_laplacian = time.time()
print("Calculating Laplacian matrix...")

from scipy import sparse

# Convert the NetworkX Laplacian matrix to a format compatible with CuPy
L_scipy = nx.laplacian_matrix(G_undirected).astype(np.float64)
print(f"Laplacian matrix (scipy format) calculated. Shape: {L_scipy.shape}")

L_cupy = csr_matrix(L_scipy)  # Convert to CuPy sparse matrix
print("Laplacian matrix converted to CuPy format.")
end_time_laplacian = time.time()
print(f"Laplacian matrix calculation and conversion completed in {end_time_laplacian - start_time_laplacian:.2f} seconds.")

# Step 7: Compute a limited number of eigenvalues and eigenvectors using GPU or fallback to CPU
try:
    print("\nStep 7: Computing a limited number of eigenvalues and eigenvectors using CUDA...")
    start_time_eigen = time.time()

    num_eigenvalues_to_compute = min(500, num_nodes)  # Compute up to 500 modes if feasible
    print(f"Number of eigenvalues to compute: {num_eigenvalues_to_compute}")

    eigenvalues, eigenvectors = eigsh(L_cupy, k=num_eigenvalues_to_compute, which='SA')

    # Convert to NumPy for compatibility
    eigenvalues = cp.asnumpy(eigenvalues)
    eigenvectors = cp.asnumpy(eigenvectors)

    end_time_eigen = time.time()
    print(f"Eigenvalue and eigenvector computation completed in {end_time_eigen - start_time_eigen:.2f} seconds.")
except cp.cuda.memory.OutOfMemoryError:
    print("Out of GPU memory. Falling back to CPU for eigenvalue computation...")
    start_time_eigen_cpu = time.time()

    eigenvalues, eigenvectors = cpu_eigsh(L_scipy, k=num_eigenvalues_to_compute, which='SM')
    end_time_eigen_cpu = time.time()
    print(f"CPU eigenvalue computation completed in {end_time_eigen_cpu - start_time_eigen_cpu:.2f} seconds.")

# Step 8: Print statistical summary of eigenvalues
print("\nStep 8: Statistical summary of computed eigenvalues...")
print(f"Number of Eigenvalues Computed: {len(eigenvalues)}")
print(f"Smallest Eigenvalue: {eigenvalues[0]:.4f}")
print(f"Largest Eigenvalue: {eigenvalues[-1]:.4f}")
print(f"Mean of Eigenvalues: {np.mean(eigenvalues):.4f}")
print(f"Standard Deviation of Eigenvalues: {np.std(eigenvalues):.4f}")

# Step 9: Apply advanced initial conditions based on high-degree nodes
print("\nStep 9: Applying advanced initial conditions...")
high_degree_nodes = sorted(G_combined.degree, key=lambda x: x[1], reverse=True)[:10]
initial_conditions = np.zeros(2 * num_eigenvalues_to_compute)
for node in high_degree_nodes:
    if node[0] in G_combined:
        idx = list(G_combined.nodes()).index(node[0])
        if idx < num_eigenvalues_to_compute:
            initial_conditions[idx] = 1  # Set initial wave amplitude at high-degree nodes

# Define the function for ODE integration
def wave_coefficients(t, y, eigenvalues):
    num_eigenvalues = len(eigenvalues)
    a = y[:num_eigenvalues]          # Coefficients a_i(t)
    a_prime = y[num_eigenvalues:]    # Coefficients a_i'(t)
    return np.concatenate([a_prime, -eigenvalues * a])

# Step 10: Solve the ODE for long-term wave propagation
print("\nStep 10: Solving ODE for long-term wave propagation...")
start_time_ode = time.time()

long_time_span = (0, 100)  # Extended time span for long-term simulation
t_eval_long = np.linspace(0, 100, 2000)
solution_long_term = solve_ivp(
    wave_coefficients,
    long_time_span,
    initial_conditions,
    t_eval=t_eval_long,
    args=(eigenvalues,)
)

end_time_ode = time.time()
print(f"ODE integration completed in {end_time_ode - start_time_ode:.2f} seconds.")
print("Long-term wave propagation simulation complete.")

# Step 11: Analyze the wave propagation results
print("\nStep 11: Statistical analysis of wave propagation results...")

# Find the maximum amplitude across all nodes at each time step
max_amplitudes_per_time_step = np.max(solution_long_term.y, axis=0)
time_of_max_amplitude = solution_long_term.t[np.argmax(max_amplitudes_per_time_step)]
overall_max_amplitude = np.max(max_amplitudes_per_time_step)

print(f"Maximum Wave Amplitude: {overall_max_amplitude:.4f}")
print(f"Time at Maximum Amplitude: {time_of_max_amplitude:.2f}")

# Summary for high-degree nodes: Print the maximum influence amplitude for select nodes
print("\nInfluence Propagation Summary for High-Degree Nodes:")
for i, node in enumerate(high_degree_nodes[:5]):
    try:
        node_idx = list(G_combined.nodes()).index(node[0])
        if node_idx < solution_long_term.y.shape[0]:  # Ensure node index is within bounds
            max_influence = np.max(solution_long_term.y[node_idx, :])
            print(f"Node {node[0]} (Degree {node[1]}): Maximum Influence Amplitude: {max_influence:.4f}")
    except IndexError:
        print(f"Node {node[0]} (Degree {node[1]}): Index out of bounds in solution data.")

print("\nFull script execution complete.")
# Write your code here :-)
