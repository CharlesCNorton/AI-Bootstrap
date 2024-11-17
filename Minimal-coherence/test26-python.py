import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
from pyts.image import RecurrencePlot
from scipy.optimize import minimize
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from skdim import id
from umap import UMAP
import tensorly as tl
from tensorly.decomposition import parafac

# Adjusted Hybrid Model Definition
def adjusted_hybrid_model(n, a, b, c, d, e):
    return (a * n) / (b + n) + c * n**2 + d + e * np.log(n + 1)

# Objective Function for Parameter Fitting
def adjusted_objective_function(params, n_values, y_values):
    a, b, c, d, e = params
    predicted_values = adjusted_hybrid_model(n_values, a, b, c, d, e)
    mse = np.mean((predicted_values - y_values) ** 2)
    return mse

# Initial Setup for Parameter Fitting
n_values_initial = np.arange(2, 8)
c_values_initial = np.array([1, 2, 5, 7, 11, 13])
initial_params_adjusted = [1, 1, 0.1, 0.1, 0.1]

# Parameter Optimization Using Minimize Function
result_adjusted = minimize(adjusted_objective_function, initial_params_adjusted, args=(n_values_initial, c_values_initial), method='Nelder-Mead')
a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted = result_adjusted.x

# Extended Range for Analysis (n = 51 to 701)
extreme_n_values = np.arange(51, 701)
extreme_coherence_values = adjusted_hybrid_model(extreme_n_values, a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted)

# Normalizing extreme_coherence_values to avoid numerical issues
normalized_coherence_values = (extreme_coherence_values - np.mean(extreme_coherence_values)) / np.std(extreme_coherence_values)

# Sliding Window Analysis for Clustering Coefficient and Connectivity
window_size = 50
clustering_coefficients = []
connected_components = []

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

plt.figure()
plt.plot(range(len(clustering_coefficients)), clustering_coefficients, label="Clustering Coefficient")
plt.xlabel("Sliding Window Index")
plt.ylabel("Clustering Coefficient")
plt.title("Sliding Window Analysis of Clustering Coefficient (n = 51 to 701)")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(connected_components)), connected_components, label="Number of Connected Components")
plt.xlabel("Sliding Window Index")
plt.ylabel("Connected Components")
plt.title("Sliding Window Analysis of Connected Components (n = 51 to 701)")
plt.legend()
plt.show()

# Step 1: Delay Embedding and Phase Space Reconstruction
embedding_dim = 5
delay = 1
embedded_data = np.array([normalized_coherence_values[i:(i + embedding_dim * delay):delay] for i in range(len(normalized_coherence_values) - embedding_dim * delay)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2])
ax.set_title("3D Phase Space Reconstruction for Extreme Coherence Values (Normalized, n = 51 to 500)")
plt.show()

# Step 2: Fractal Dimension Calculation using Correlation Dimension
def correlation_dimension(data, max_radius):
    distances = pdist(data)
    radii = np.linspace(0, max_radius, 50)
    counts = [np.sum(distances < r) for r in radii]
    log_r = np.log(radii[1:])
    log_C = np.log(counts[1:])
    slope, _ = np.polyfit(log_r, log_C, 1)
    return slope

corr_dim = correlation_dimension(embedded_data, max_radius=100)
print(f"Estimated Correlation Dimension of Phase Space: {corr_dim}")

# Step 3: Advanced Dimensionality Reduction (t-SNE and UMAP)
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_result = tsne.fit_transform(embedded_data)
plt.figure()
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis')
plt.title("t-SNE Projection of Embedded Coherence Data (n = 51 to 500)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

umap = UMAP(n_components=2, random_state=0)
umap_result = umap.fit_transform(embedded_data)
plt.figure()
plt.scatter(umap_result[:, 0], umap_result[:, 1], cmap='viridis')
plt.title("UMAP Projection of Embedded Coherence Data (n = 51 to 500)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Step 4: Tensor Decomposition to Detect Latent Structures
tensor_data = tl.tensor(embedded_data)
tensor_factors = parafac(tensor_data, rank=3)
print("Tensor Decomposition Factors (Rank 3):")
for factor in tensor_factors:
    print(factor)

# Step 5: Recurrence Quantification Analysis (RQA)
rp = RecurrencePlot(threshold='point', percentage=20)
coherence_rp = rp.fit_transform(normalized_coherence_values.reshape(1, -1))
plt.figure(figsize=(8, 8))
plt.imshow(coherence_rp[0], cmap='binary', origin='lower')
plt.title("Recurrence Plot for Extreme Coherence Values (n = 51 to 500)")
plt.show()

# Step 6: Graph Analysis for Emergent Symmetry
threshold = 20
G = nx.Graph()
for i in range(len(extreme_n_values)):
    G.add_node(extreme_n_values[i])
    for j in range(i + 1, len(extreme_n_values)):
        if abs(extreme_coherence_values[i] - extreme_coherence_values[j]) < threshold:
            G.add_edge(extreme_n_values[i], extreme_n_values[j])

symmetry_properties = {
    "Number of Nodes": G.number_of_nodes(),
    "Number of Edges": G.number_of_edges(),
    "Clustering Coefficient": nx.average_clustering(G),
    "Number of Connected Components": nx.number_connected_components(G),
    "Graph Automorphisms": len(list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter()))
}
print("Graph Symmetry Analysis in Extreme Dimensions (n = 51 to 701):")
for key, value in symmetry_properties.items():
    print(f"{key}: {value}")

# Step 7: Fractal Behavior and Strange Attractors Analysis
dim_estimator = id.CorrInt()
fractal_dimension_estimation = dim_estimator.fit(embedded_data).dimension_
print(f"Estimated Fractal Dimension using Correlation Integral: {fractal_dimension_estimation}")
