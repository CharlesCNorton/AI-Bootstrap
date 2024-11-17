import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def compute_C_n(n):
    if n >= 2 and n in [2, 3]:
        return n - 1
    elif n in [4, 5]:
        return 2 * n - 3
    elif n >= 6:
        return 2 * n - 1
    else:
        raise ValueError("n must be an integer greater than or equal to 2")

def generate_C_sequence(n_values):
    C_values = []
    for n in n_values:
        try:
            C = compute_C_n(n)
            C_values.append(C)
            logging.info(f"C({n}) = {C}")
        except ValueError as e:
            logging.warning(e)
    return np.array(C_values)

def perform_linear_regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = LinearRegression()
    reg.fit(x, y)
    y_pred = reg.predict(x)
    r2 = r2_score(y, y_pred)
    logging.info(f"Linear Regression: slope = {reg.coef_[0]}, intercept = {reg.intercept_}, R^2 = {r2}")
    return {
        'slope': reg.coef_[0],
        'intercept': reg.intercept_,
        'r_squared': r2,
        'predictions': y_pred
    }

def detect_phase_transitions(n_values, C_values):
    differences = np.diff(C_values)
    transitions = np.where(np.diff(differences) != 0)[0] + 1
    logging.info(f"Phase transitions detected at indices: {transitions}")
    return transitions

def compute_topological_properties(C_values):
    G = nx.Graph()
    nodes = list(range(len(C_values)))
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    clustering_coeff = nx.average_clustering(G)
    try:
        avg_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        avg_path_length = float('inf')
        logging.warning("Graph is not connected; average path length is infinite.")
    logging.info(f"Clustering Coefficient: {clustering_coeff}")
    logging.info(f"Average Path Length: {avg_path_length}")
    return {
        'clustering_coefficient': clustering_coeff,
        'average_path_length': avg_path_length
    }

def compute_self_similarity(C_values):
    C_values = np.array(C_values)
    mean_C = np.mean(C_values)
    autocorrelation = np.correlate(C_values - mean_C, C_values - mean_C, mode='full')
    autocorrelation = autocorrelation[autocorrelation.size // 2:]
    autocorrelation /= autocorrelation[0]
    entropy = -np.sum((C_values / np.sum(C_values)) * np.log2(C_values / np.sum(C_values)))
    logging.info(f"Entropy of the sequence: {entropy}")
    return {
        'autocorrelation': autocorrelation,
        'entropy': entropy
    }

def compute_statistical_invariants(C_values):
    C_values = np.array(C_values)
    threshold = 0.1 * np.std(C_values)
    distance_matrix = squareform(pdist(C_values.reshape(-1, 1)))
    recurrence_matrix = (distance_matrix < threshold).astype(int)
    recurrence_rate = np.sum(recurrence_matrix) / (len(C_values) ** 2)
    logging.info(f"Recurrence Rate: {recurrence_rate}")
    def higuchi_fd(X, kmax):
        N = len(X)
        Lk = []
        for k in range(1, kmax+1):
            Lmk = []
            for m in range(k):
                Lm = 0
                n_max = int(np.floor((N - m - 1) / k))
                if n_max < 1:
                    continue
                for i in range(1, n_max):
                    Lm += abs(X[m + i * k] - X[m + (i - 1) * k])
                norm = (N - 1) / (k * n_max * k)
                if norm == 0:
                    continue
                Lm = (Lm * norm)
                Lmk.append(Lm)
            if len(Lmk) == 0:
                Lk.append(np.nan)
            else:
                Lk.append(np.mean(Lmk))
        Lk = np.array(Lk)
        valid_indices = ~np.isnan(Lk) & (Lk > 0)
        lnLk = np.log(Lk[valid_indices])
        lnk = np.log(np.arange(1, kmax+1)[valid_indices])
        if len(lnk) < 2:
            logging.warning("Not enough data points to compute fractal dimension.")
            fd = np.nan
        else:
            fd, _ = np.polyfit(lnk, lnLk, 1)
        return fd
    fractal_dimension = higuchi_fd(C_values, kmax=5)
    logging.info(f"Fractal Dimension: {fractal_dimension}")
    return {
        'recurrence_rate': recurrence_rate,
        'fractal_dimension': fractal_dimension
    }

def analyze_number_theoretic_properties(C_values):
    C_values = np.array(C_values)
    parity = C_values % 2
    modulo_patterns = {}
    for m in range(2, 6):
        modulo_pattern = C_values % m
        modulo_patterns[m] = modulo_pattern
        logging.info(f"Modulo {m} pattern: {modulo_pattern}")
    return {
        'parity': parity,
        'modulo_patterns': modulo_patterns
    }

def analyze_structural_stability(C_values):
    C_values = np.array(C_values)
    first_diff = np.diff(C_values)
    second_diff = np.diff(first_diff)
    stable_phase_indices = [i for i, n in enumerate(range(2, len(C_values)+2)) if n >= 6]
    stable_phase = C_values[stable_phase_indices]
    stable_first_diff = np.diff(stable_phase)
    stability = np.all(stable_first_diff == stable_first_diff[0])
    logging.info(f"Structural stability in the stable phase: {stability}")
    return {
        'first_differences': first_diff,
        'second_differences': second_diff,
        'stability': stability
    }

def main_analysis(n_values, C_values):
    results = {}
    results['linear_regression'] = perform_linear_regression(n_values, C_values)
    results['phase_transitions'] = detect_phase_transitions(n_values, C_values)
    results['topological_properties'] = compute_topological_properties(C_values)
    results['self_similarity'] = compute_self_similarity(C_values)
    results['statistical_invariants'] = compute_statistical_invariants(C_values)
    results['number_theoretic_properties'] = analyze_number_theoretic_properties(C_values)
    results['structural_stability'] = analyze_structural_stability(C_values)
    return results

if __name__ == "__main__":
    n_values = list(range(2, 11))
    C_values = generate_C_sequence(n_values)
    results = main_analysis(n_values, C_values)
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"{subkey}: {subvalue}")
        else:
            print(value)
