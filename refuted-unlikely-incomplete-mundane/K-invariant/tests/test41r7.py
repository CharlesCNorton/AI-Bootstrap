import numpy as np
from gudhi import RipsComplex
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy import isprime
from itertools import combinations
import networkx as nx
from scipy.stats import linregress, ks_2samp
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class SymmetryHierarchyAnalyzer:
    def __init__(self, num_points=100, max_dim=5, num_trials=50):
        self.num_points = num_points
        self.max_dim = max_dim
        self.num_trials = num_trials
        self.primes = [p for p in range(3, 23) if isprime(p)]
        self.symmetry_groups = self._initialize_symmetry_groups()

    def _initialize_symmetry_groups(self):
        """Initialize known symmetry groups with their orders"""
        return {
            'cyclic': [p for p in self.primes],
            'dihedral': [2*p for p in self.primes],
            'tetrahedral': [12],
            'octahedral': [24],
            'icosahedral': [60],
            'binary': [24, 48, 120]
        }

    def compute_enhanced_k_invariant(self, points, include_noise_test=True):
        """Compute enhanced K-invariant with all components"""
        rips = RipsComplex(points=points, max_edge_length=1.0)
        st = rips.create_simplex_tree(max_dimension=self.max_dim)
        persistence = st.persistence()

        # Extract persistence by dimension
        dim_lifetimes = {d: [] for d in range(self.max_dim + 1)}
        for dim, (birth, death) in persistence:
            if death != float('inf'):
                dim_lifetimes[dim].append(death - birth)

        # Compute basic components
        components = {
            'log_term': sum(np.log1p(l)**2 for lifetimes in dim_lifetimes.values()
                           for l in lifetimes),
            'cross_term': self._compute_cross_term(dim_lifetimes),
            'geometric': np.mean(pdist(points)),
            'periodic': sum(np.sin(np.pi * l / 2) for lifetimes in dim_lifetimes.values()
                           for l in lifetimes),
            'curvature': self._estimate_curvature(points),
            'symmetry': self._compute_symmetry_term(dim_lifetimes),
            'torsion': self._compute_torsion_term(dim_lifetimes),
            'stability': self._compute_stability_measure(dim_lifetimes),
            'nullspace': self._compute_nullspace_dimension(points)
        }

        # Add noise resistance only if requested (to prevent recursion)
        if include_noise_test:
            components['noise_resistance'] = self._compute_noise_resistance(points)

        return components, dim_lifetimes

    def _sample_sphere(self, dim):
        """Sample points from n-sphere"""
        points = np.random.normal(0, 1, (self.num_points, dim+1))
        return points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    def _generate_product_space(self, dim1, dim2):
        """Generate product space S^dim1 × S^dim2"""
        s1 = self._sample_sphere(dim1)
        s2 = self._sample_sphere(dim2)
        return np.column_stack([s1, s2])

    def _generate_cp2(self):
        """Generate complex projective space CP²"""
        points = np.random.normal(0, 1, (self.num_points, 3)) + \
                1j * np.random.normal(0, 1, (self.num_points, 3))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return np.column_stack([points.real, points.imag])

    def _generate_quaternionic_space(self):
        """Generate quaternionic projective space HP¹"""
        points = np.random.normal(0, 1, (self.num_points, 4)) + \
                1j * np.random.normal(0, 1, (self.num_points, 4))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return np.column_stack([points.real, points.imag])

    def _generate_lens_space(self, p, q):
        """Generate lens space L(p,q)"""
        points = self._sample_sphere(3)
        theta = 2 * np.pi * q / p
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        for i in range(len(points)):
            z1 = points[i,0:2]
            z2 = points[i,2:4]
            z1 = rotation @ z1
            points[i,0:2] = z1
            points[i,2:4] = z2
        return points

    def _generate_quotient_space(self, order, group_type):
        """Generate quotient space with specified symmetry group"""
        points = self._sample_sphere(3)
        theta = 2 * np.pi / order
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

        for i in range(len(points)):
            z1 = points[i,0:2]
            z2 = points[i,2:4]
            if group_type == 'cyclic':
                z1 = rotation @ z1
            elif group_type == 'binary':
                z1 = rotation @ z1
                z2 = rotation @ z2
            points[i,0:2] = z1
            points[i,2:4] = z2

        return points

    def _compute_cross_term(self, dim_lifetimes):
        """Compute cross-term interactions"""
        cross_term = 0
        for dim, lifetimes in dim_lifetimes.items():
            if len(lifetimes) > 1:
                lifetimes = np.array(lifetimes)
                cross_term += sum(abs(l1 * l2) for i, l1 in enumerate(lifetimes)
                                for l2 in lifetimes[i+1:])
        return cross_term

    def _compute_symmetry_term(self, dim_lifetimes):
        """Compute term sensitive to symmetries"""
        return sum(np.sin(2 * np.pi * l * p) / p
                  for p in self.primes
                  for lifetimes in dim_lifetimes.values()
                  for l in lifetimes)

    def _compute_torsion_term(self, dim_lifetimes):
        """Compute term sensitive to torsion"""
        return sum(np.sin(2 * np.pi * l) * np.cos(np.pi * l)
                  for lifetimes in dim_lifetimes.values()
                  for l in lifetimes)

    def _estimate_curvature(self, points):
        """Estimate local curvature"""
        if len(points) < 4:
            return 0
        dist_matrix = squareform(pdist(points))
        k = min(20, len(points)-1)
        curvatures = []
        for i in range(len(points)):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            neighbor_points = points[neighbors]
            center = np.mean(neighbor_points, axis=0)
            radii = np.linalg.norm(neighbor_points - center, axis=1)
            curvatures.append(1/np.mean(radii))
        return np.mean(curvatures)

    def _compute_stability_measure(self, dim_lifetimes):
        """Compute stability measure for persistence features"""
        all_lifetimes = [l for lifetimes in dim_lifetimes.values() for l in lifetimes]
        if not all_lifetimes:
            return 0
        return np.std(all_lifetimes) / (np.mean(all_lifetimes) + 1e-10)

    def _compute_noise_resistance(self, points, noise_level=0.01):
        """Compute noise resistance measure without recursion"""
        noisy_points = points + np.random.normal(0, noise_level, points.shape)
        noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1)[:, np.newaxis]

        original_components, _ = self.compute_enhanced_k_invariant(points, include_noise_test=False)
        noisy_components, _ = self.compute_enhanced_k_invariant(noisy_points, include_noise_test=False)

        differences = []
        for k in ['log_term', 'cross_term', 'geometric', 'periodic', 'curvature']:
            if original_components[k] != 0:
                diff = abs(original_components[k] - noisy_components[k]) / original_components[k]
                differences.append(diff)

        return np.mean(differences) if differences else 1.0

    def _compute_nullspace_dimension(self, points):
        """Compute approximate nullspace dimension"""
        try:
            _, s, _ = np.linalg.svd(points)
            return sum(s < 1e-10)
        except:
            return 0

    def generate_test_spaces(self, include_random=True):
        """Generate test spaces with known symmetry properties"""
        spaces = {}

        # Standard spaces
        spaces.update({
            'S2': self._sample_sphere(2),
            'S3': self._sample_sphere(3),
            'S2xS2': self._generate_product_space(2, 2),
            'CP2': self._generate_cp2(),
            'HP1': self._generate_quaternionic_space()
        })

        # Lens spaces for each prime
        for p in self.primes:
            for q in range(1, (p+1)//2 + 1):
                if np.gcd(p, q) == 1:
                    spaces[f'L({p},{q})'] = self._generate_lens_space(p, q)

        # Quotient spaces with various symmetry groups
        for group_type, orders in self.symmetry_groups.items():
            for order in orders:
                spaces[f'{group_type}_{order}'] = self._generate_quotient_space(order, group_type)

        # Control spaces
        if include_random:
            for i in range(5):
                spaces[f'Random_{i}'] = np.random.normal(0, 1, (self.num_points, 4))
                spaces[f'Random_{i}'] = spaces[f'Random_{i}'] / np.linalg.norm(spaces[f'Random_{i}'], axis=1)[:, np.newaxis]

        return spaces

    def analyze_symmetry_hierarchy(self):
        """Main analysis method"""
        results = {}

        # Generate and analyze spaces
        spaces = self.generate_test_spaces()
        print("Computing invariants and performing validation...")

        for name, points in tqdm(spaces.items()):
            trial_results = []

            for _ in range(self.num_trials):
                perturbed_points = points + np.random.normal(0, 0.01, points.shape)
                perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1)[:, np.newaxis]

                components, _ = self.compute_enhanced_k_invariant(perturbed_points)
                trial_results.append(components)

            # Compute statistics across trials
            results[name] = {
                'mean': {k: np.mean([t[k] for t in trial_results]) for k in trial_results[0].keys()},
                'std': {k: np.std([t[k] for t in trial_results]) for k in trial_results[0].keys()},
                'cv': {k: np.std([t[k] for t in trial_results])/np.mean([t[k] for t in trial_results])
                      if np.mean([t[k] for t in trial_results]) != 0 else 0
                      for k in trial_results[0].keys()}
            }

            # Add validation metrics
            results[name]['validation'] = {
                'stability': np.mean([t['stability'] for t in trial_results]),
                'noise_resistance': np.mean([t['noise_resistance'] for t in trial_results]),
                'nullspace': np.mean([t['nullspace'] for t in trial_results])
            }

        return results

def main():
    analyzer = SymmetryHierarchyAnalyzer(num_points=100, max_dim=5, num_trials=50)

    print("Starting comprehensive analysis with validation...")
    results = analyzer.analyze_symmetry_hierarchy()

    print("\nResults Summary:")
    for space_name, space_results in results.items():
        print(f"\n{space_name}:")
        print("Component Values (mean ± std [CV]):")
        for component, value in space_results['mean'].items():
            std = space_results['std'][component]
            cv = space_results['cv'][component]
            print(f"  {component}: {value:.4f} ± {std:.4f} [{cv:.4f}]")

        print("Validation Metrics:")
        for metric, value in space_results['validation'].items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
