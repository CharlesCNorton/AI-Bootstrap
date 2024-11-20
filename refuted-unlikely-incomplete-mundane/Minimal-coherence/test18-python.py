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

def generate_sequence_data(max_n=100):
    """Generate comprehensive sequence data"""
    values = [C(n) for n in range(2, max_n)]
    diffs = [np.diff(values, n) for n in range(1, 6)]
    ratios = [values[i+1]/values[i] for i in range(len(values)-1)]

    return {
        'values': values,
        'diffs': diffs,
        'ratios': ratios,
        'log_values': np.log(values),
        'normalized': (values - np.mean(values)) / np.std(values)
    }

def detailed_modular_analysis(max_n=50, max_modulus=12):
    """Enhanced modular pattern analysis"""
    values = [C(n) for n in range(2, max_n)]
    modular_patterns = {}

    for mod in range(2, max_modulus+1):
        residues = [v % mod for v in values]
        pattern = []
        for i in range(mod):
            pattern.append(residues.count(i))

        # Advanced period detection
        periods = []
        for period_length in range(1, len(residues)//2):
            chunks = [tuple(residues[i:i+period_length])
                     for i in range(0, len(residues)-period_length, period_length)]
            if len(set(chunks)) == 1:
                periods.append(period_length)

        # Analyze residue transitions
        transition_matrix = np.zeros((mod, mod))
        for i in range(len(residues)-1):
            transition_matrix[residues[i], residues[i+1]] += 1

        # Calculate pattern metrics
        entropy = stats.entropy(pattern)
        uniformity = 1 - np.std(pattern)/np.mean(pattern) if np.mean(pattern) != 0 else 0

        modular_patterns[mod] = {
            'residues': residues[:20],  # First 20 values
            'distribution': pattern,
            'periods': periods,
            'min_period': min(periods) if periods else None,
            'entropy': entropy,
            'uniformity': uniformity,
            'transition_matrix': transition_matrix
        }

    return modular_patterns

def advanced_symmetry_analysis(max_n=50):
    """Comprehensive symmetry analysis"""
    values = [C(n) for n in range(2, max_n)]

    # Local symmetry detection with classification
    local_symmetries = []
    for window in range(3, 10):
        for i in range(len(values)-window):
            segment = values[i:i+window]
            rev_segment = segment[::-1]

            # Test different symmetry types
            tests = {
                'reflection': np.allclose(segment, rev_segment),
                'translation': np.allclose(np.diff(segment), np.diff(rev_segment)),
                'scaling': np.allclose(np.diff(np.log(segment)), -np.diff(np.log(rev_segment))),
                'arithmetic': np.allclose(np.diff(np.diff(segment)), 0),
                'geometric': np.allclose(np.diff(np.log(segment)), np.mean(np.diff(np.log(segment))))
            }

            if any(tests.values()):
                local_symmetries.append({
                    'position': i,
                    'window': window,
                    'types': {k: v for k, v in tests.items() if v}
                })

    # Global symmetry analysis
    diffs = np.diff(values)
    global_patterns = {
        'arithmetic': np.std(np.diff(diffs)) < 0.1,
        'geometric': np.std(np.diff(np.log(values))) < 0.1,
        'fibonacci_like': np.corrcoef(values[:-2],
                                    [values[i+2]-values[i+1] for i in range(len(values)-2)])[0,1],
        'self_similarity': analyze_self_similarity(values)
    }

    return {
        'local_symmetries': local_symmetries,
        'global_patterns': global_patterns,
        'symmetry_points': len(local_symmetries),
        'symmetry_distribution': analyze_symmetry_distribution(local_symmetries)
    }

def analyze_self_similarity(values):
    """Analyze self-similarity patterns"""
    similarities = []
    for window in range(2, len(values)//2):
        chunks = [values[i:i+window] for i in range(0, len(values)-window, window)]
        if len(chunks) > 1:
            correlations = [np.corrcoef(chunks[i], chunks[i+1])[0,1]
                          for i in range(len(chunks)-1)]
            similarities.append(np.mean(correlations))
    return similarities

def analyze_symmetry_distribution(symmetries):
    """Analyze the distribution of symmetry points"""
    if not symmetries:
        return None

    positions = [s['position'] for s in symmetries]
    windows = [s['window'] for s in symmetries]

    return {
        'position_stats': {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'clusters': KMeans(n_clusters=min(3, len(positions))).fit(np.array(positions).reshape(-1, 1)).labels_
        },
        'window_stats': {
            'mean': np.mean(windows),
            'std': np.std(windows),
            'most_common': max(set(windows), key=windows.count)
        }
    }

def phase_space_analysis(max_n=50):
    """Analyze sequence in phase space"""
    values = [C(n) for n in range(2, max_n)]

    # Create phase space embeddings
    embeddings = {}
    for dim in range(2, 5):
        embedding = []
        for i in range(len(values)-dim+1):
            embedding.append(values[i:i+dim])
        embeddings[dim] = np.array(embedding)

    # Analyze phase space properties
    phase_properties = {}
    for dim, embedding in embeddings.items():
        # Calculate distances between points
        distances = pdist(embedding)

        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(embedding)

        phase_properties[dim] = {
            'variance_ratio': pca.explained_variance_ratio_,
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'distance_std': np.std(distances),
            'dimensionality': sum(pca.explained_variance_ratio_ > 0.01)  # Effective dimensionality
        }

    return phase_properties

def algebraic_invariant_analysis(max_n=50):
    """Analyze algebraic invariants of the sequence"""
    values = [C(n) for n in range(2, max_n)]

    invariants = {}

    # Test various algebraic combinations
    for k in range(1, 5):
        # Linear combinations
        linear_combs = []
        for i in range(len(values)-k):
            comb = sum(values[i:i+k])
            linear_combs.append(comb)

        # Multiplicative combinations
        mult_combs = []
        for i in range(len(values)-k):
            comb = np.prod(values[i:i+k])
            mult_combs.append(comb)

        # Polynomial combinations
        poly_combs = []
        for i in range(len(values)-k):
            coeffs = np.polyfit(range(k), values[i:i+k], k-1)
            poly_combs.append(coeffs[0])

        invariants[k] = {
            'linear': {
                'mean': np.mean(linear_combs),
                'std': np.std(linear_combs),
                'ratio': np.mean(np.diff(linear_combs))
            },
            'multiplicative': {
                'mean': np.mean(mult_combs),
                'std': np.std(mult_combs),
                'ratio': np.mean(np.diff(mult_combs))
            },
            'polynomial': {
                'mean': np.mean(poly_combs),
                'std': np.std(poly_combs),
                'stability': np.std(poly_combs)/np.mean(poly_combs) if np.mean(poly_combs) != 0 else float('inf')
            }
        }

    return invariants

def structural_transition_analysis():
    """Analyze structural transitions in detail"""
    values = [C(n) for n in range(2, 30)]
    transitions = []

    # Multiple detection methods
    methods = {
        'difference': lambda x: np.diff(x),
        'ratio': lambda x: np.diff(np.log(x)),
        'acceleration': lambda x: np.diff(np.diff(x)),
        'curvature': lambda x: np.diff(np.diff(np.diff(x)))
    }

    transition_points = defaultdict(list)
    for method_name, method in methods.items():
        signal = method(np.array(values))
        peaks, _ = find_peaks(np.abs(signal), height=np.std(signal))
        transition_points[method_name] = peaks + 2  # Adjust for indexing

    # Analyze each detected transition
    all_transitions = set()
    for transitions in transition_points.values():
        all_transitions.update(transitions)

    transition_analysis = {}
    for t in sorted(all_transitions):
        if t < len(values)-2:
            before = values[max(0, t-3):t]
            after = values[t:t+3]

            transition_analysis[t] = {
                'before_pattern': np.polyfit(range(len(before)), before, 1)[0],
                'after_pattern': np.polyfit(range(len(after)), after, 1)[0],
                'discontinuity': after[0] - before[-1],
                'pattern_change': np.mean(np.diff(after)) / np.mean(np.diff(before)) if len(before) > 1 and len(after) > 1 else float('inf'),
                'detection_methods': [m for m, ts in transition_points.items() if t in ts]
            }

    return transition_analysis

def mega_comprehensive_analysis():
    """Enhanced comprehensive analysis combining all approaches"""
    results = {}

    print("\nINITIATING MEGA-COMPREHENSIVE ANALYSIS")
    print("=====================================")

    # 1. Modular Analysis
    print("\nPerforming modular analysis...")
    results['modular'] = detailed_modular_analysis()

    # 2. Symmetry Analysis
    print("Analyzing symmetries...")
    results['symmetry'] = advanced_symmetry_analysis()

    # 3. Phase Space Analysis
    print("Analyzing phase space properties...")
    results['phase_space'] = phase_space_analysis()

    # 4. Algebraic Invariants
    print("Computing algebraic invariants...")
    results['invariants'] = algebraic_invariant_analysis()

    # 5. Structural Transitions
    print("Analyzing structural transitions...")
    results['transitions'] = structural_transition_analysis()

    # Print comprehensive results
    print("\nANALYSIS RESULTS")
    print("===============")

    print("\n1. Modular Properties:")
    for mod, data in results['modular'].items():
        if data['min_period']:
            print(f"Modulo {mod}: Min Period = {data['min_period']}, Entropy = {data['entropy']:.3f}")

    print("\n2. Symmetry Properties:")
    sym_results = results['symmetry']
    print(f"Total symmetry points: {sym_results['symmetry_points']}")
    print("Global patterns:", sym_results['global_patterns'])

    print("\n3. Phase Space Properties:")
    for dim, props in results['phase_space'].items():
        print(f"\nDimension {dim}:")
        print(f"Effective dimensionality: {props['dimensionality']}")
        print(f"Variance explained: {props['variance_ratio']}")

    print("\n4. Algebraic Invariants:")
    for k, invs in results['invariants'].items():
        print(f"\nOrder {k}:")
        print(f"Linear stability: {invs['linear']['std']:.3f}")
        print(f"Multiplicative stability: {invs['multiplicative']['std']:.3f}")

    print("\n5. Structural Transitions:")
    for point, analysis in results['transitions'].items():
        print(f"\nTransition at n={point}:")
        print(f"Pattern change ratio: {analysis['pattern_change']:.3f}")
        print(f"Detected by methods: {analysis['detection_methods']}")

    return results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mega_results = mega_comprehensive_analysis()
