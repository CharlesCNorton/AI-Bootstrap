import numpy as np
from gudhi import RipsComplex
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class InvariantAnalyzer:
    def __init__(self, num_points=100, max_dim=5, num_trials=50):
        self.num_points = num_points
        self.max_dim = max_dim
        self.num_trials = num_trials

    def generate_test_spaces(self):
        """Generate test spaces with controlled geometric and topological properties"""
        spaces = {}

        # 1. Topologically equivalent, geometrically different
        spaces['sphere_round'] = self._sample_sphere(2)
        spaces['sphere_stretched'] = self._sample_stretched_sphere(2, 2.0)
        spaces['sphere_squashed'] = self._sample_stretched_sphere(2, 0.5)

        # 2. Geometrically similar, topologically different
        # Ensure each space gets the full number of points
        spaces['torus_standard'] = self._sample_torus(1, 0.5)
        spaces['double_torus'] = self._sample_connected_tori(2, 1, 0.5)
        spaces['triple_torus'] = self._sample_connected_tori(3, 1, 0.5)

        # 3. Random control spaces
        spaces['random_uniform'] = self._normalize(np.random.uniform(-1, 1, (self.num_points, 3)))
        spaces['random_normal'] = self._normalize(np.random.normal(0, 1, (self.num_points, 3)))
        spaces['random_sphere'] = self._sample_sphere(2)

        # 4. Spaces with known homotopy groups
        spaces['S2'] = self._sample_sphere(2)
        spaces['S3'] = self._sample_sphere(3)
        spaces['CP2'] = self._generate_cp2()

        return spaces

    def _normalize(self, points):
        """Normalize points to unit norm"""
        return points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    def _sample_sphere(self, dim, noise=0.0):
        """Sample points from n-sphere with optional noise"""
        points = np.random.normal(0, 1, (self.num_points, dim+1))
        points = self._normalize(points)
        if noise > 0:
            points += np.random.normal(0, noise, points.shape)
            points = self._normalize(points)
        return points

    def _sample_stretched_sphere(self, dim, stretch_factor):
        """Sample points from a sphere with one axis stretched"""
        points = self._sample_sphere(dim)
        points[:, 0] *= stretch_factor
        return points

    def _sample_torus(self, R, r):
        """Sample points from a torus"""
        theta = 2 * np.pi * np.random.random(self.num_points)
        phi = 2 * np.pi * np.random.random(self.num_points)

        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)

        return np.column_stack([x, y, z])

    def _sample_connected_tori(self, num_tori, R, r):
        """Sample points from multiple connected tori"""
        points_per_torus = self.num_points // num_tori
        remainder = self.num_points % num_tori

        all_points = []
        current_offset = 0

        for i in range(num_tori):
            # Add extra point from remainder if needed
            current_points = points_per_torus + (1 if i < remainder else 0)

            # Generate torus points
            theta = 2 * np.pi * np.random.random(current_points)
            phi = 2 * np.pi * np.random.random(current_points)

            x = (R + r * np.cos(phi)) * np.cos(theta) + current_offset
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)

            all_points.append(np.column_stack([x, y, z]))
            current_offset += 2*R  # Offset for next torus

        return np.vstack(all_points)

    def _generate_cp2(self):
        """Generate points in CP²"""
        points = np.random.normal(0, 1, (self.num_points, 3)) + \
                1j * np.random.normal(0, 1, (self.num_points, 3))
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return np.column_stack([points.real, points.imag])

    def compute_invariants(self, points):
        """Compute both geometric and topological invariants"""
        # Geometric invariants
        geometric_invariants = {
            'mean_distance': np.mean(pdist(points)),
            'std_distance': np.std(pdist(points)),
            'max_distance': np.max(pdist(points)),
            'volume': np.prod(np.max(points, axis=0) - np.min(points, axis=0)),
            'curvature': self._estimate_curvature(points)
        }

        # Topological invariants
        rips = RipsComplex(points=points, max_edge_length=2.0)
        st = rips.create_simplex_tree(max_dimension=self.max_dim)
        persistence = st.persistence()

        topological_invariants = {
            'betti_numbers': st.betti_numbers(),
            'persistence_entropy': self._compute_persistence_entropy(persistence),
            'persistence_landscape': self._compute_persistence_landscape(persistence),
            'bottleneck_distance': self._compute_bottleneck_distance(persistence)
        }

        return geometric_invariants, topological_invariants

    def _estimate_curvature(self, points):
        """Estimate local curvature using k-nearest neighbors"""
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

    def _compute_persistence_entropy(self, persistence):
        """Compute persistence entropy"""
        if not persistence:
            return 0

        lifetimes = np.array([death - birth for _, (birth, death) in persistence
                            if death != float('inf')])
        if len(lifetimes) == 0:
            return 0

        normalized_lifetimes = lifetimes / np.sum(lifetimes)
        entropy = -np.sum(normalized_lifetimes * np.log(normalized_lifetimes + 1e-10))
        return entropy

    def _compute_persistence_landscape(self, persistence):
        """Compute first persistence landscape"""
        if not persistence:
            return np.zeros(10)

        lifetimes = np.array([death - birth for _, (birth, death) in persistence
                            if death != float('inf')])
        if len(lifetimes) == 0:
            return np.zeros(10)

        t = np.linspace(0, np.max(lifetimes) if len(lifetimes) > 0 else 1, 10)
        landscape = np.zeros_like(t)

        for i, ti in enumerate(t):
            values = np.maximum(0, np.minimum(ti - lifetimes, lifetimes))
            landscape[i] = np.max(values) if len(values) > 0 else 0

        return landscape

    def _compute_bottleneck_distance(self, persistence1, persistence2=None):
        """Compute bottleneck distance to empty diagram"""
        if not persistence1:
            return 0

        lifetimes = np.array([death - birth for _, (birth, death) in persistence1
                            if death != float('inf')])
        if len(lifetimes) == 0:
            return 0

        return np.max(lifetimes)

    def analyze_spaces(self):
        """Perform comprehensive analysis of test spaces"""
        spaces = self.generate_test_spaces()
        results = {}

        print("Computing invariants for all spaces...")
        for name, points in tqdm(spaces.items()):
            geometric, topological = self.compute_invariants(points)
            results[name] = {
                'geometric': geometric,
                'topological': topological,
                'points': points
            }

        # Analyze correlations
        correlations = self._analyze_correlations(results)

        # Analyze geometric vs topological separation
        separation = self._analyze_separation(results)

        # Analyze random space behavior
        random_analysis = self._analyze_random_behavior(results)

        return results, correlations, separation, random_analysis

    def _analyze_correlations(self, results):
        """Analyze correlations between invariants"""
        invariant_names = []
        invariant_values = []

        for space_name, space_results in results.items():
            values = []
            names = []

            # Geometric invariants
            for name, value in space_results['geometric'].items():
                if isinstance(value, (int, float)):
                    values.append(value)
                    names.append(f'geo_{name}')

            # Topological invariants
            for name, value in space_results['topological'].items():
                if isinstance(value, (int, float)):
                    values.append(value)
                    names.append(f'top_{name}')
                elif isinstance(value, np.ndarray) and len(value.shape) == 1:
                    values.extend(value)
                    names.extend([f'top_{name}_{i}' for i in range(len(value))])

            if not invariant_names:
                invariant_names = names
            invariant_values.append(values)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(np.array(invariant_values).T)

        return {
            'names': invariant_names,
            'matrix': corr_matrix
        }

    def _analyze_separation(self, results):
        """Analyze separation between geometric and topological features"""
        geometric_features = []
        topological_features = []
        labels = []

        for space_name, space_results in results.items():
            geo_vector = []
            top_vector = []

            # Geometric features
            for name, value in space_results['geometric'].items():
                if isinstance(value, (int, float)):
                    geo_vector.append(value)

            # Topological features
            for name, value in space_results['topological'].items():
                if isinstance(value, (int, float)):
                    top_vector.append(value)
                elif isinstance(value, np.ndarray) and len(value.shape) == 1:
                    top_vector.extend(value)

            geometric_features.append(geo_vector)
            topological_features.append(top_vector)
            labels.append(space_name)

        # Perform PCA
        pca_geo = PCA(n_components=2).fit_transform(geometric_features)
        pca_top = PCA(n_components=2).fit_transform(topological_features)

        return {
            'geometric_pca': pca_geo,
            'topological_pca': pca_top,
            'labels': labels
        }

    def _analyze_random_behavior(self, results):
        """Analyze why random spaces show similar behavior"""
        random_spaces = ['random_uniform', 'random_normal', 'random_sphere']
        structured_spaces = [name for name in results.keys() if name not in random_spaces]

        distributions = {
            'geometric': {},
            'topological': {}
        }

        for space_type in ['random', 'structured']:
            spaces = random_spaces if space_type == 'random' else structured_spaces

            for space_name in spaces:
                for inv_type in ['geometric', 'topological']:
                    for name, value in results[space_name][inv_type].items():
                        if isinstance(value, (int, float)):
                            if name not in distributions[inv_type]:
                                distributions[inv_type][name] = {'random': [], 'structured': []}
                            distributions[inv_type][name][space_type].append(value)

        # Compute KS statistics
        ks_stats = {
            'geometric': {},
            'topological': {}
        }

        for inv_type in ['geometric', 'topological']:
            for name, dist in distributions[inv_type].items():
                if len(dist['random']) > 0 and len(dist['structured']) > 0:
                    stat, pval = ks_2samp(dist['random'], dist['structured'])
                    ks_stats[inv_type][name] = {'statistic': stat, 'pvalue': pval}

        return {
            'distributions': distributions,
            'ks_stats': ks_stats
        }

    def visualize_results(self, results, correlations, separation, random_analysis):
        """Create comprehensive visualizations"""
        plt.figure(figsize=(20, 15))

        # 1. Correlation heatmap
        plt.subplot(221)
        sns.heatmap(correlations['matrix'],
                   xticklabels=correlations['names'],
                   yticklabels=correlations['names'])
        plt.title('Correlation between Invariants')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # 2. Geometric vs Topological PCA
        plt.subplot(222)
        for i, label in enumerate(separation['labels']):
            plt.scatter(separation['geometric_pca'][i, 0],
                       separation['geometric_pca'][i, 1],
                       label=label)
        plt.title('PCA of Geometric Features')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Random vs Structured Distributions
        plt.subplot(223)
        x_pos = np.arange(len(random_analysis['ks_stats']['geometric']))
        for inv_type in ['geometric', 'topological']:
            stats = [v['statistic'] for v in random_analysis['ks_stats'][inv_type].values()]
            plt.bar(x_pos + (0.4 if inv_type == 'topological' else 0),
                   stats,
                   width=0.4,
                   label=inv_type)
            x_pos = np.arange(len(stats))
        plt.title('KS Statistics: Random vs Structured')
        plt.xticks(x_pos + 0.4, list(random_analysis['ks_stats']['geometric'].keys()), rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    analyzer = InvariantAnalyzer(num_points=100, max_dim=5, num_trials=50)
    results, correlations, separation, random_analysis = analyzer.analyze_spaces()

    print("\nCorrelation Analysis:")
    high_correlations = []
    for i, name1 in enumerate(correlations['names']):
        for j, name2 in enumerate(correlations['names']):
            if i < j and abs(correlations['matrix'][i,j]) > 0.9:
                high_correlations.append((name1, name2, correlations['matrix'][i,j]))

    print("\nHighly correlated invariants (|r| > 0.9):")
    for name1, name2, corr in high_correlations:
        print(f"{name1} - {name2}: {corr:.3f}")

    print("\nRandom vs Structured Analysis:")
    for inv_type in ['geometric', 'topological']:
        print(f"\n{inv_type.capitalize()} Invariants:")
        for name, stats in random_analysis['ks_stats'][inv_type].items():
            print(f"{name}: KS stat = {stats['statistic']:.3f}, p-value = {stats['pvalue']:.3f}")

    analyzer.visualize_results(results, correlations, separation, random_analysis)

if __name__ == "__main__":
    main()
