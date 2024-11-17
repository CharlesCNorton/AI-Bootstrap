import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict
import math
from itertools import combinations
from scipy.signal import find_peaks
from scipy.fft import fft
import networkx as nx

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

def deep_sequence_analysis(max_n=100):
    """Comprehensive sequence analysis including patterns, symmetries, and algebraic properties"""
    values = [C(n) for n in range(2, max_n)]

    # FFT Analysis
    fft_vals = np.abs(fft(values))
    dominant_freqs = find_peaks(fft_vals)[0]

    # Pattern Detection
    diffs = [np.diff(values, n) for n in range(1, 6)]

    # Algebraic Structure
    differences = defaultdict(list)
    ratios = defaultdict(list)
    for k in range(1, 6):
        for i in range(len(values)-k):
            differences[k].append(values[i+k] - values[i])
            if values[i] != 0:
                ratios[k].append(values[i+k] / values[i])

    return {
        'fft': {
            'dominant_frequencies': dominant_freqs,
            'spectrum': fft_vals
        },
        'differences': differences,
        'ratios': ratios,
        'raw_values': values
    }

def modular_analysis(max_n=50, max_modulus=12):
    """Detailed analysis of modular patterns"""
    values = [C(n) for n in range(2, max_n)]
    modular_patterns = {}

    for mod in range(2, max_modulus+1):
        residues = [v % mod for v in values]
        pattern = []
        for i in range(mod):
            pattern.append(residues.count(i))

        # Find pattern period
        for period in range(1, len(residues)//2):
            if residues[period:2*period] == residues[:period]:
                break
        else:
            period = None

        modular_patterns[mod] = {
            'residues': residues,
            'distribution': pattern,
            'period': period,
            'entropy': stats.entropy(pattern)
        }

    return modular_patterns

def symmetry_analysis():
    """Advanced symmetry detection"""
    values = [C(n) for n in range(2, 30)]

    # Local symmetries
    local_sym = []
    for window in range(3, 8):
        for i in range(len(values)-window):
            segment = values[i:i+window]
            rev_segment = segment[::-1]
            if np.allclose(np.diff(segment), -np.diff(rev_segment)):
                local_sym.append((i, window))

    # Global symmetries
    diffs = np.diff(values)
    global_patterns = {
        'arithmetic': np.std(np.diff(diffs)) < 0.1,
        'geometric': np.std(np.diff(np.log(values))) < 0.1,
        'fibonacci_like': np.corrcoef(values[:-2],
                                    [values[i+2]-values[i+1] for i in range(len(values)-2)])[0,1]
    }

    return {
        'local_symmetries': local_sym,
        'global_patterns': global_patterns
    }

def graph_theoretic_analysis():
    """Analyze sequence as a graph structure"""
    values = [C(n) for n in range(2, 30)]

    # Create difference graph
    G = nx.Graph()
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            G.add_edge(i, j, weight=values[j]-values[i])

    # Analyze graph properties
    analysis = {
        'degree_distribution': list(dict(G.degree()).values()),
        'clustering': nx.average_clustering(G),
        'path_lengths': nx.average_shortest_path_length(G),
        'centrality': nx.degree_centrality(G)
    }

    return analysis

def statistical_tests():
    """Comprehensive statistical analysis"""
    values = [C(n) for n in range(2, 50)]
    diffs = np.diff(values)

    # Fit various distributions
    distributions = [
        stats.norm, stats.poisson, stats.gamma, stats.exponweib
    ]

    fit_results = {}
    for dist in distributions:
        try:
            params = dist.fit(diffs)
            ks_stat = stats.kstest(diffs, dist.name, params)
            fit_results[dist.name] = {
                'params': params,
                'ks_stat': ks_stat
            }
        except:
            continue

    return fit_results

def comprehensive_mega_analysis():
    """Master analysis combining all approaches"""
    results = {}

    # 1. Deep Sequence Analysis
    results['sequence'] = deep_sequence_analysis()

    # 2. Modular Patterns
    results['modular'] = modular_analysis()

    # 3. Symmetries
    results['symmetry'] = symmetry_analysis()

    # 4. Graph Theory
    results['graph'] = graph_theoretic_analysis()

    # 5. Statistical Tests
    results['stats'] = statistical_tests()

    # Print comprehensive results
    print("\nMEGA-ANALYSIS RESULTS")
    print("=====================")

    print("\n1. Sequence Properties:")
    print(f"Dominant frequencies: {results['sequence']['fft']['dominant_frequencies']}")
    print(f"First-order differences stability: {np.std(results['sequence']['differences'][1]):.3f}")

    print("\n2. Modular Patterns:")
    for mod, data in results['modular'].items():
        if data['period']:
            print(f"Modulo {mod}: Period = {data['period']}, Entropy = {data['entropy']:.3f}")

    print("\n3. Symmetry Analysis:")
    print(f"Local symmetry points: {len(results['symmetry']['local_symmetries'])}")
    print("Global patterns:", results['symmetry']['global_patterns'])

    print("\n4. Graph Properties:")
    print(f"Average clustering: {results['graph']['clustering']:.3f}")
    print(f"Average path length: {results['graph']['path_lengths']:.3f}")

    print("\n5. Statistical Distribution Fits:")
    for dist, fit in results['stats'].items():
        if 'ks_stat' in fit:
            print(f"{dist}: KS test p-value = {fit['ks_stat'].pvalue:.6f}")

    return results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mega_results = comprehensive_mega_analysis()
