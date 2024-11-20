import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
from pyts.image import RecurrencePlot
from scipy.optimize import minimize

def adjusted_hybrid_model(n, a, b, c, d, e):
    return (a * n) / (b + n) + c * n**2 + d + e * np.log(n + 1)

def adjusted_objective_function(params, n_values, y_values):
    a, b, c, d, e = params
    predicted_values = adjusted_hybrid_model(n_values, a, b, c, d, e)
    mse = np.mean((predicted_values - y_values) ** 2)
    return mse

n_values_initial = np.arange(2, 8)
c_values_initial = np.array([1, 2, 5, 7, 11, 13])
initial_params_adjusted = [1, 1, 0.1, 0.1, 0.1]

result_adjusted = minimize(adjusted_objective_function, initial_params_adjusted, args=(n_values_initial, c_values_initial), method='Nelder-Mead')

a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted = result_adjusted.x

extreme_n_values = np.arange(101, 151)
extreme_coherence_values = adjusted_hybrid_model(extreme_n_values, a_adjusted, b_adjusted, c_adjusted, d_adjusted, e_adjusted)

topological_data = np.array([extreme_coherence_values]).reshape(-1, 1)
ph_results = ripser(topological_data, maxdim=1)

plt.figure(figsize=(10, 6))
plot_diagrams(ph_results['dgms'], show=True)
plt.title("Persistence Diagrams for Extreme Coherence Values (n = 101 to 150)")
plt.show()

rp = RecurrencePlot(threshold='point', percentage=20)
coherence_rp = rp.fit_transform(extreme_coherence_values.reshape(1, -1))

plt.figure(figsize=(8, 8))
plt.imshow(coherence_rp[0], cmap='binary', origin='lower')
plt.title("Recurrence Plot for Extreme Coherence Values (n = 101 to 150)")
plt.show()

threshold = 15
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

print("Graph Symmetry Analysis in Extreme Dimensions (n = 101 to 150):")
for key, value in symmetry_properties.items():
    print(f"{key}: {value}")
