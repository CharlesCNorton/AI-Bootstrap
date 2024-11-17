import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict
import math
from itertools import combinations, product
from scipy.signal import find_peaks
from scipy.fft import fft
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.cluster.hierarchy import dendrogram, linkage

def C(n):
    """Calculate minimal coherence conditions for n-categories"""
    if n < 2:
        raise ValueError("n must be >= 2")

    if n <= 3:
        return n - 1  # Foundational phase
    elif n <= 5:
        return 2*n - 3  # Transitional phase
    else:
        return 2*n - 1  # Linear phase

def calculate_fractal_dimension(values, min_eps=0.1, max_eps=2.0, num_points=20):
    """Calculate fractal dimension using box-counting method"""
    points = np.column_stack((range(len(values)), values))

    # Normalize points to [0,1] x [0,1]
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

    # Generate logarithmically spaced epsilon values
    eps_values = np.logspace(np.log10(min_eps), np.log10(max_eps), num_points)
    counts = []

    for eps in eps_values:
        boxes = set()
        for point in points:
            box_x = math.floor(point[0]/eps)
            box_y = math.floor(point[1]/eps)
            boxes.add((box_x, box_y))
        counts.append(len(boxes))

    # Fit line to log-log plot
    coeffs = np.polyfit(np.log(1/eps_values), np.log(counts), 1)
    return coeffs[0], np.std(np.log(counts))

def recurrence_analysis(values, threshold=0.1):
    """Perform recurrence quantification analysis"""
    # Normalize values
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())

    N = len(values)
    dist_matrix = squareform(pdist(values.reshape(-1, 1)))
    rec_matrix = dist_matrix < threshold

    # Calculate RQA measures
    recurrence_rate = np.sum(rec_matrix) / (N**2)

    # Calculate diagonal lines
    diag_lengths = []
    for i in range(-N+1, N):
        diag = np.diag(rec_matrix, k=i)
        current_length = 0
        for point in diag:
            if point:
                current_length += 1
            elif current_length > 0:
                diag_lengths.append(current_length)
                current_length = 0

    if diag_lengths:
        determinism = np.sum([l for l in diag_lengths if l > 1]) / np.sum(diag_lengths)
        avg_diag_length = np.mean(diag_lengths)
        max_diag_length = max(diag_lengths)
    else:
        determinism = 0
        avg_diag_length = 0
        max_diag_length = 0

    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diagonal_length': avg_diag_length,
        'max_diagonal_length': max_diag_length
    }

def topological_analysis(values, max_dim=3):
    """Perform topological data analysis"""
    # Normalize values
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())

    # Create point cloud with proper dimensions
    points = []
    for i in range(len(values) - max_dim + 1):
        point = [values[i + j] for j in range(max_dim)]
        points.append(point)

    points = np.array(points)

    if len(points) < 2:
        return {
            'connected_components': 1,
            'clustering_coefficient': 0,
            'average_path_length': 0,
            'dimension_estimate': 1
        }

    # Calculate persistent homology approximation
    nbrs = NearestNeighbors(n_neighbors=min(10, len(points))).fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Create graph from distance matrix
    threshold = np.mean(distances)
    G = nx.Graph()

    # Add nodes first
    for i in range(len(points)):
        G.add_node(i)

    # Add edges
    for i in range(len(points)):
        for j, dist in zip(indices[i], distances[i]):
            if dist < threshold:
                G.add_edge(i, j, weight=dist)

    # Analyze topological features
    topology = {
        'connected_components': nx.number_connected_components(G),
        'clustering_coefficient': nx.average_clustering(G),
        'average_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
        'dimension_estimate': np.log(len(G.edges()) + 1) / np.log(len(G.nodes())) if len(G.nodes()) > 0 else 0
    }

    return topology

def analyze_self_similarity_detailed(values, max_window=20):
    """Detailed analysis of self-similarity patterns"""
    # Normalize values
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())

    results = {}

    for window in range(2, min(max_window, len(values)//2)):
        chunks = [values[i:i+window] for i in range(0, len(values)-window+1)]

        # Calculate similarity matrix
        sim_matrix = np.zeros((len(chunks), len(chunks)))
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks):
                sim_matrix[i,j] = np.corrcoef(chunk1, chunk2)[0,1]

        # Analyze similarity distribution
        results[window] = {
            'mean_similarity': np.mean(sim_matrix),
            'std_similarity': np.std(sim_matrix),
            'max_similarity': np.max(sim_matrix[sim_matrix < 1]) if np.any(sim_matrix < 1) else 1,
            'similarity_entropy': entropy(sim_matrix.flatten())
        }

    return results

def phase_space_reconstruction(values, embedding_dim=3, delay=1):
    """Reconstruct phase space using time-delay embedding"""
    # Normalize values
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())

    N = len(values)
    if N < embedding_dim * delay:
        return None

    embedded = np.zeros((N - (embedding_dim-1)*delay, embedding_dim))

    for i in range(embedded.shape[0]):
        for j in range(embedding_dim):
            embedded[i,j] = values[i + j*delay]

    # Calculate phase space properties
    properties = {
        'volume': np.prod(np.max(embedded, axis=0) - np.min(embedded, axis=0)),
        'density': len(embedded) / np.prod(np.max(embedded, axis=0) - np.min(embedded, axis=0)),
        'correlation_dim': np.mean([np.corrcoef(embedded[:,i], embedded[:,j])[0,1]
                                  for i, j in combinations(range(embedding_dim), 2)])
    }

    return properties

def ultimate_mega_analysis(max_n=100):
    """Comprehensive mega-analysis with all approaches"""
    print("\nINITIATING ULTIMATE MEGA-ANALYSIS")
    print("================================")

    values = [C(n) for n in range(2, max_n)]
    results = {}

    try:
        # 1. Fractal Analysis
        print("\nPerforming fractal analysis...")
        results['fractal'] = calculate_fractal_dimension(values)

        # 2. Recurrence Analysis
        print("Performing recurrence analysis...")
        results['recurrence'] = recurrence_analysis(values)

        # 3. Topological Analysis
        print("Performing topological analysis...")
        results['topology'] = topological_analysis(values)

        # 4. Detailed Self-Similarity
        print("Analyzing self-similarity patterns...")
        results['self_similarity'] = analyze_self_similarity_detailed(values)

        # 5. Phase Space Reconstruction
        print("Reconstructing phase space...")
        results['phase_space'] = phase_space_reconstruction(values)

        # Print comprehensive results
        print("\nULTIMATE ANALYSIS RESULTS")
        print("========================")

        print("\n1. Fractal Properties:")
        print(f"Fractal dimension: {results['fractal'][0]:.3f} ± {results['fractal'][1]:.3f}")

        print("\n2. Recurrence Properties:")
        for key, value in results['recurrence'].items():
            print(f"{key}: {value:.3f}")

        print("\n3. Topological Features:")
        for key, value in results['topology'].items():
            print(f"{key}: {value if isinstance(value, int) else f'{value:.3f}'}")

        print("\n4. Self-Similarity Analysis:")
        window_stats = results['self_similarity'][5]  # Example window size
        print(f"Window size 5 statistics:")
        for key, value in window_stats.items():
            print(f"{key}: {value:.3f}")

        print("\n5. Phase Space Properties:")
        if results['phase_space']:
            for key, value in results['phase_space'].items():
                print(f"{key}: {value:.3f}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

    return results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ultimate_results = ultimate_mega_analysis()
