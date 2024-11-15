import numpy as np
from gudhi import RipsComplex
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy import isprime
from itertools import combinations
import networkx as nx
from scipy.stats import linregress
from scipy.optimize import curve_fit

class LensSpaceAnalyzer:
    def __init__(self, num_points=100, max_dim=5):
        self.num_points = num_points
        self.max_dim = max_dim
        self.primes = [p for p in range(3, 23) if isprime(p)]  # Test first few prime numbers

    def generate_lens_space(self, p, q):
        """Generate points in lens space L(p,q)"""
        points = np.random.normal(0, 1, (self.num_points, 4))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

        # Apply lens space identification
        theta = 2 * np.pi * q / p
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

        for i in range(self.num_points):
            z1 = points[i,0:2]
            z2 = points[i,2:4]
            z1 = rotation @ z1
            points[i,0:2] = z1
            points[i,2:4] = z2
        return points

    def generate_test_spaces(self):
        """Generate comprehensive test suite of spaces"""
        spaces = {}

        # Generate lens spaces for all primes
        for p in self.primes:
            for q in range(1, (p+1)//2 + 1):  # Only need to test up to (p+1)/2
                if np.gcd(p, q) == 1:  # Only valid lens spaces
                    spaces[f'L({p},{q})'] = self.generate_lens_space(p, q)

        # Generate comparison spaces
        spaces.update({
            'CP2': self._generate_cp2(),
            'S2xS1': self._generate_product_space(2, 1),
            'S2xS2': self._generate_product_space(2, 2),
            'S3': self._sample_sphere(3),
            'HP1': self._generate_quaternionic_projective_space(),
            'G24': self._generate_binary_tetrahedral_space()
        })

        return spaces

    def _generate_cp2(self):
        """Generate complex projective space CP²"""
        points = np.random.normal(0, 1, (self.num_points, 3)) + \
                1j * np.random.normal(0, 1, (self.num_points, 3))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return np.column_stack([points.real, points.imag])

    def _sample_sphere(self, dim):
        """Sample points from n-sphere"""
        points = np.random.normal(0, 1, (self.num_points, dim+1))
        return points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    def _generate_product_space(self, dim1, dim2):
        """Generate product space"""
        s1 = self._sample_sphere(dim1)
        s2 = self._sample_sphere(dim2)
        return np.column_stack([s1, s2])

    def _generate_quaternionic_projective_space(self):
        """Generate HP¹ for comparison"""
        points = np.random.normal(0, 1, (self.num_points, 4)) + \
                1j * np.random.normal(0, 1, (self.num_points, 4))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return np.column_stack([points.real, points.imag])

    def _generate_binary_tetrahedral_space(self):
        """Generate binary tetrahedral space"""
        points = self._sample_sphere(3)
        # Apply binary tetrahedral group action
        theta = 2 * np.pi / 24
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        for i in range(self.num_points):
            z1 = points[i,0:2]
            z2 = points[i,2:4]
            z1 = rotation @ z1
            points[i,0:2] = z1
            points[i,2:4] = z2
        return points

    def compute_enhanced_k_invariant(self, points):
        """Compute enhanced K-invariant with all components"""
        rips = RipsComplex(points=points, max_edge_length=1.0)
        st = rips.create_simplex_tree(max_dimension=self.max_dim)
        persistence = st.persistence()

        # Extract persistence by dimension
        dim_lifetimes = {d: [] for d in range(self.max_dim + 1)}
        for dim, (birth, death) in persistence:
            if death != float('inf'):
                dim_lifetimes[dim].append(death - birth)

        # Compute components
        components = {
            'log_term': self._compute_log_term(dim_lifetimes),
            'cross_term': self._compute_cross_term(dim_lifetimes),
            'geometric': np.mean(pdist(points)),
            'periodic': self._compute_periodic_term(dim_lifetimes),
            'curvature': self._estimate_curvature(points),
            'torsion': self._compute_torsion_term(dim_lifetimes),
            'prime_sensitive': self._compute_prime_sensitive_term(dim_lifetimes)
        }

        return components, dim_lifetimes

    def _compute_log_term(self, dim_lifetimes):
        return sum(np.log1p(l)**2 for lifetimes in dim_lifetimes.values() for l in lifetimes)

    def _compute_cross_term(self, dim_lifetimes):
        cross_term = 0
        for dim, lifetimes in dim_lifetimes.items():
            if len(lifetimes) > 1:
                lifetimes = np.array(lifetimes)
                cross_term += sum(abs(l1 * l2) for i, l1 in enumerate(lifetimes)
                                for l2 in lifetimes[i+1:])
        return cross_term

    def _compute_periodic_term(self, dim_lifetimes):
        return sum(np.sin(np.pi * l / 2) for lifetimes in dim_lifetimes.values()
                  for l in lifetimes)

    def _compute_torsion_term(self, dim_lifetimes):
        """Compute term sensitive to torsion"""
        return sum(np.sin(2 * np.pi * l) * np.cos(np.pi * l)
                  for lifetimes in dim_lifetimes.values() for l in lifetimes)

    def _compute_prime_sensitive_term(self, dim_lifetimes):
        """Compute term potentially sensitive to prime order symmetries"""
        return sum(np.sin(np.pi * l * p) / p
                  for p in self.primes
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

    def analyze_prime_relationships(self):
        """Analyze relationships between spaces with focus on prime-order behavior"""
        spaces = self.generate_test_spaces()
        results = {}

        print("Computing invariants for all spaces...")
        for name, points in tqdm(spaces.items()):
            components, lifetimes = self.compute_enhanced_k_invariant(points)
            results[name] = {
                'components': components,
                'lifetimes': lifetimes
            }

        # Analyze prime-order patterns
        prime_patterns = self._analyze_prime_patterns(results)

        # Look for sqrt-like relationships
        sqrt_relationships = self._find_sqrt_relationships(results)

        # Analyze scaling with prime order
        prime_scaling = self._analyze_prime_scaling(results)

        return {
            'results': results,
            'prime_patterns': prime_patterns,
            'sqrt_relationships': sqrt_relationships,
            'prime_scaling': prime_scaling
        }

    def _analyze_prime_patterns(self, results):
        """Analyze patterns related to prime orders"""
        patterns = {}

        # For each prime p
        for p in self.primes:
            lens_spaces = [name for name in results.keys() if f'L({p},' in name]

            # Compare with non-lens spaces
            non_lens = [name for name in results.keys() if 'L(' not in name]

            for space in non_lens:
                for lens in lens_spaces:
                    ratio = self._compute_ratio(results[space], results[lens])
                    patterns[f'{space}/{lens}'] = ratio

        return patterns

    def _find_sqrt_relationships(self, results):
        """Look for sqrt-like relationships"""
        sqrt_relations = []

        for space1, space2 in combinations(results.keys(), 2):
            ratio = self._compute_ratio(results[space1], results[space2])

            # Check for sqrt-like ratios
            for p in self.primes:
                if any(abs(r - np.sqrt(p)) < 0.1 for r in ratio.values()):
                    sqrt_relations.append((space1, space2, p, ratio))

        return sqrt_relations

    def _analyze_prime_scaling(self, results):
        """Analyze how invariants scale with prime order"""
        scaling = {}

        # For each component
        for component in next(iter(results.values()))['components'].keys():
            prime_values = []

            # Collect values for each prime
            for p in self.primes:
                lens_spaces = [name for name in results.keys() if f'L({p},' in name]
                if lens_spaces:
                    avg_value = np.mean([results[name]['components'][component]
                                       for name in lens_spaces])
                    prime_values.append((p, avg_value))

            # Fit scaling relationship
            if prime_values:
                x = np.array([v[0] for v in prime_values])
                y = np.array([v[1] for v in prime_values])

                # Try different scaling relationships
                scaling[component] = self._fit_scaling_relationships(x, y)

        return scaling

    def _fit_scaling_relationships(self, x, y):
        """Fit various scaling relationships"""
        fits = {}

        # Linear scaling
        linear = linregress(x, y)
        fits['linear'] = {'slope': linear.slope, 'r_value': linear.rvalue}

        # Power law scaling
        try:
            def power_law(x, a, b):
                return a * x**b
            popt, _ = curve_fit(power_law, x, y)
            fits['power_law'] = {'exponent': popt[1]}
        except:
            fits['power_law'] = None

        # Logarithmic scaling
        try:
            log_fit = linregress(np.log(x), y)
            fits['logarithmic'] = {'slope': log_fit.slope, 'r_value': log_fit.rvalue}
        except:
            fits['logarithmic'] = None

        return fits

    def _compute_ratio(self, result1, result2):
        """Compute ratio between components of two spaces"""
        ratios = {}
        for component in result1['components'].keys():
            if result2['components'][component] != 0:
                ratios[component] = (result1['components'][component] /
                                   result2['components'][component])
        return ratios

    def visualize_results(self, analysis_results):
        """Create comprehensive visualizations"""
        plt.figure(figsize=(20, 15))

        # Plot 1: Prime scaling relationships
        plt.subplot(221)
        self._plot_prime_scaling(analysis_results['prime_scaling'])

        # Plot 2: Sqrt relationship distribution
        plt.subplot(222)
        self._plot_sqrt_relationships(analysis_results['sqrt_relationships'])

        # Plot 3: Component correlation network
        plt.subplot(223)
        self._plot_component_network(analysis_results['results'])

        # Plot 4: Prime pattern heatmap
        plt.subplot(224)
        self._plot_prime_patterns(analysis_results['prime_patterns'])

        plt.tight_layout()
        plt.show()

    def _plot_prime_scaling(self, scaling):
        """Plot scaling relationships with prime order"""
        for component, fits in scaling.items():
            if 'power_law' in fits and fits['power_law']:
                plt.plot(self.primes,
                        [fits['power_law']['exponent'] * p for p in self.primes],
                        label=f'{component} (power={fits["power_law"]["exponent"]:.2f})')
        plt.xlabel('Prime Order')
        plt.ylabel('Component Value')
        plt.title('Scaling with Prime Order')
        plt.legend()

    def _plot_sqrt_relationships(self, sqrt_relations):
        """Plot distribution of sqrt-like relationships"""
        if sqrt_relations:
            ratios = [list(r[3].values()) for r in sqrt_relations]
            plt.hist(np.concatenate(ratios), bins=50)
            plt.xlabel('Ratio Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of √p-like Ratios')

    def _plot_component_network(self, results):
        """Plot network of component relationships"""
        G = nx.Graph()

        # Add nodes for spaces
        for space in results.keys():
            G.add_node(space)

        # Add edges for significant relationships
        for space1, space2 in combinations(results.keys(), 2):
            ratio = self._compute_ratio(results[space1], results[space2])
            for component, r in ratio.items():
                if any(abs(r - np.sqrt(p)) < 0.1 for p in self.primes):
                    G.add_edge(space1, space2, weight=r)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8)
        plt.title('Component Relationship Network')

    def _plot_prime_patterns(self, patterns):
        """Plot heatmap of prime-related patterns"""
        if patterns:
            matrix = np.array([[v for v in d.values()] for d in patterns.values()])
            plt.imshow(matrix, aspect='auto', cmap='viridis')
            plt.colorbar(label='Ratio Value')
            plt.title('Prime Pattern Heatmap')

def main():
    analyzer = LensSpaceAnalyzer(num_points=100, max_dim=5)
    print("Starting comprehensive analysis of prime-order lens spaces...")

    analysis_results = analyzer.analyze_prime_relationships()

    print("\nSignificant √p Relationships Found:")
    for space1, space2, p, ratio in analysis_results['sqrt_relationships']:
        for component, r in ratio.items():
            if abs(r - np.sqrt(p)) < 0.1:
                print(f"{space1}/{space2} ({component}): {r:.4f} ≈ √{p}")

    print("\nPrime Scaling Patterns:")
    for component, scaling in analysis_results['prime_scaling'].items():
        if 'power_law' in scaling and scaling['power_law']:
            print(f"{component}: scales as p^{scaling['power_law']['exponent']:.4f}")

    analyzer.visualize_results(analysis_results)

if __name__ == "__main__":
    main()
