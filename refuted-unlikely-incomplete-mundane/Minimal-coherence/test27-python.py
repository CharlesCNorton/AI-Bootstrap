import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
from pyts.image import RecurrencePlot
from scipy.optimize import minimize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import sympy as sp
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from skdim import id
from sympy.combinatorics import Permutation, PermutationGroup
import scipy.stats as stats

# ---- Adjusted Hybrid Model ----
def adjusted_hybrid_model(n, a, b, c, d, e):
    return (a * n) / (b + n) + c * n**2 + d + e * np.log(n + 1)

# ---- Objective Function for Parameter Fitting ----
def adjusted_objective_function(params, n_values, y_values):
    a, b, c, d, e = params
    predicted_values = adjusted_hybrid_model(n_values, a, b, c, d, e)
    mse = np.mean((predicted_values - y_values) ** 2)
    return mse

# ---- Initial Setup for Parameter Fitting ----
n_values_initial = np.arange(2, 8)
c_values_initial = np.array([1, 2, 5, 7, 11, 13])
initial_params_adjusted = [1, 1, 0.1, 0.1, 0.1]

# ---- Parameter Optimization ----
result_adjusted = minimize(adjusted_objective_function, initial_params_adjusted, args=(n_values_initial, c_values_initial), method='Nelder-Mead')
a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted = result_adjusted.x

# ---- Extended Range for Analysis (n = 51 to 500) ----
extreme_n_values = np.arange(51, 501)
extreme_coherence_values = adjusted_hybrid_model(extreme_n_values, a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted)

# Normalizing coherence values
normalized_coherence_values = (extreme_coherence_values - np.mean(extreme_coherence_values)) / np.std(extreme_coherence_values)

# ---- Sliding Window Analysis for Clustering Coefficient and Connected Components ----
window_size = 50
clustering_coefficients = []
connected_components = []
average_shortest_path_lengths = []
average_degrees = []

for start in range(0, len(extreme_n_values) - window_size + 1, window_size):
    G = nx.Graph()
    end = start + window_size
    sub_n_values = extreme_n_values[start:end]
    sub_coherence_values = extreme_coherence_values[start:end]

    for i in range(len(sub_n_values)):
        G.add_node(sub_n_values[i])
        for j in range(i + 1, len(sub_n_values)):
            if abs(sub_coherence_values[i] - sub_coherence_values[j]) < 20:
                G.add_edge(sub_n_values[i], sub_n_values[j])

    clustering_coefficients.append(nx.average_clustering(G))
    connected_components.append(nx.number_connected_components(G))
    if nx.is_connected(G):
        average_shortest_path_lengths.append(nx.average_shortest_path_length(G))
    else:
        average_shortest_path_lengths.append(np.nan)
    average_degrees.append(np.mean([degree for node, degree in G.degree()]))

# Plot Sliding Window Statistics
plt.figure()
plt.plot(range(len(clustering_coefficients)), clustering_coefficients, label="Clustering Coefficient")
plt.xlabel("Sliding Window Index")
plt.ylabel("Clustering Coefficient")
plt.title("Sliding Window Analysis of Clustering Coefficient (n = 51 to 500)")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(connected_components)), connected_components, label="Number of Connected Components")
plt.xlabel("Sliding Window Index")
plt.ylabel("Connected Components")
plt.title("Sliding Window Analysis of Connected Components (n = 51 to 500)")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(average_shortest_path_lengths)), average_shortest_path_lengths, label="Average Shortest Path Length")
plt.xlabel("Sliding Window Index")
plt.ylabel("Average Shortest Path Length")
plt.title("Sliding Window Analysis of Shortest Path Length (n = 51 to 500)")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(average_degrees)), average_degrees, label="Average Degree")
plt.xlabel("Sliding Window Index")
plt.ylabel("Average Degree")
plt.title("Sliding Window Analysis of Average Degree (n = 51 to 500)")
plt.legend()
plt.show()

# ---- Hamiltonian Modeling: Coherence as Integrable System ----
n = sp.symbols('n')
a, b, c, d, e = sp.symbols('a b c d e')
H = (a * n) / (b + n) + c * n**2 + d + e * sp.log(n + 1)
dn_dt = sp.diff(H, n)
print("Equation of Motion for n:", dn_dt)
conserved_quantity = sp.integrate(dn_dt, n)
print("Potential Conserved Quantity:", conserved_quantity)

# ---- Tensor Decomposition (Rank 3 and Tucker) ----
embedding_dim = 5
delay = 1
embedded_data = np.array([normalized_coherence_values[i:(i + embedding_dim * delay):delay] for i in range(len(normalized_coherence_values) - embedding_dim * delay)])
tensor_data = tl.tensor(embedded_data)

# Corrected Tucker Decomposition rank keyword
tucker_decomposition = tucker(tensor_data, rank=[3, 3])
print("Tucker Decomposition Core Tensor and Factors:")
print("Core Tensor:", tucker_decomposition.core)
print("Factors:", tucker_decomposition.factors)

# ---- Advanced Dimensionality Reduction ----
# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_result = tsne.fit_transform(embedded_data)
plt.figure()
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis')
plt.title("t-SNE Projection of Embedded Coherence Data (n = 51 to 500)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# UMAP
umap = UMAP(n_components=2, random_state=0)
umap_result = umap.fit_transform(embedded_data)
plt.figure()
plt.scatter(umap_result[:, 0], umap_result[:, 1], cmap='viridis')
plt.title("UMAP Projection of Embedded Coherence Data (n = 51 to 500)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# ---- Kernel PCA for Non-linear Relationships ----
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
kpca_result = kpca.fit_transform(embedded_data)
plt.figure()
plt.scatter(kpca_result[:, 0], kpca_result[:, 1], cmap='viridis')
plt.title("Kernel PCA Projection of Embedded Coherence Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# ---- Graph Automorphism Analysis ----
G = nx.Graph()
for i in range(len(extreme_n_values)):
    G.add_node(extreme_n_values[i])
    for j in range(i + 1, len(extreme_n_values)):
        if abs(extreme_coherence_values[i] - extreme_coherence_values[j]) < 20:
            G.add_edge(extreme_n_values[i], extreme_n_values[j])

# Extract Automorphisms
GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
automorphisms = list(GM.isomorphisms_iter())

# Create a mapping from nodes to sequential indices (0, 1, 2, ...)
node_list = list(G.nodes)
node_to_index = {node: index for index, node in enumerate(node_list)}

# Corrected Extraction to Handle Node Keys Properly
# Extract Permutations for Each Automorphism Using Node Mapping
permutations = []
for automorphism in automorphisms:
    # Map each node in the automorphism to its corresponding index
    mapping = [node_to_index[automorphism[node]] for node in node_list]
    permutations.append(Permutation(mapping))

# Create Group Representation
automorphism_group = PermutationGroup(permutations)
print("Order of Automorphism Group:", automorphism_group.order())
print("Is Group Abelian?", automorphism_group.is_abelian)
print("Generators of Automorphism Group:", automorphism_group.generators)

# ---- Persistence Homology (Sliding Windows) ----
window_size = 100
step_size = 50

for start in range(0, len(embedded_data) - window_size + 1, step_size):
    end = start + window_size
    window_data = embedded_data[start:end, :]
    ph_results = ripser(window_data, maxdim=1)
    plt.figure(figsize=(10, 6))
    plot_diagrams(ph_results['dgms'], show=True)
    plt.title(f"Persistence Diagrams for Sliding Window {start}-{end} (n = 51 to 500)")
    plt.show()

# ---- Fractal Behavior and Strange Attractors Analysis ----
dim_estimator = id.CorrInt()
fractal_dimension_estimation = dim_estimator.fit(embedded_data).dimension_
print(f"Estimated Fractal Dimension using Correlation Integral: {fractal_dimension_estimation}")

# ---- Additional Statistical Insights ----
# Distribution Analysis: Normality Test
shapiro_test_stat, shapiro_p_value = stats.shapiro(normalized_coherence_values)
print(f"Shapiro-Wilk Normality Test p-value: {shapiro_p_value}")

# Correlation Analysis between Clustering Coefficient and Average Degree
corr_coefficient, corr_p_value = stats.pearsonr(clustering_coefficients, average_degrees[:len(clustering_coefficients)])
print(f"Pearson Correlation between Clustering Coefficient and Average Degree: {corr_coefficient}, p-value: {corr_p_value}")
