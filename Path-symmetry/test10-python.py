import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import itertools
from scipy.special import factorial
import networkx as nx

@dataclass
class CategoryData:
    dimension: int
    known_coherences: int
    theoretical_max: int
    observed_structures: List[str]

@dataclass
class CoherenceResult:
    dimension: int
    R_bound: int
    S_bound: int
    T_bound: int
    predicted_coherences: int
    actual_coherences: Optional[int]
    stability_metric: float

def create_path_space(x, y, dim):
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * dim)
    perturbation = np.random.uniform(-1, 1, (dim, dim))
    return np.eye(dim) + epsilon * np.exp(-0.3 * dist) * perturbation

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

class CoherenceReductionAnalyzer:
    def __init__(self, base_analyzer):
        self.base = base_analyzer
        self.reduction_patterns = {}
        self.bottleneck_structure = {}

    def analyze_reduction_mechanism(self, results):
        reduction_data = {
            'factors': [],
            'log_factors': [],
            'dimension_ratios': [],
            'cancellation_patterns': {},
            'models': {
                'exponential': lambda x, a, b: a * np.exp(-b * x),
                'power_law': lambda x, a, b: a * x**(-b),
                'log_linear': lambda x, a, b: a * x * np.log(x) + b
            }
        }

        for dim in range(2, 8):
            if dim in self.base.known_categories:
                predicted = results[dim-2].predicted_coherences
                actual = self.base.known_categories[dim].known_coherences
                factor = predicted / actual
                log_factor = np.log(factor)

                reduction_data['factors'].append(factor)
                reduction_data['log_factors'].append(log_factor)
                reduction_data['dimension_ratios'].append(factor / dim)

                theoretical_max = self.base.theoretical_coherence_bound(dim)
                actual_reduction = theoretical_max / actual
                reduction_data['cancellation_patterns'][dim] = {
                    'theoretical_max': theoretical_max,
                    'actual': actual,
                    'reduction_ratio': actual_reduction,
                    'efficiency': np.log2(actual_reduction)
                }

        x = np.arange(2, 8)
        log_factors = np.array(reduction_data['log_factors'])

        model_fits = {}
        for name, model in reduction_data['models'].items():
            try:
                popt, pcov = curve_fit(model, x, log_factors)
                residuals = log_factors - model(x, *popt)
                r_squared = 1 - (np.sum(residuals**2) / np.sum((log_factors - np.mean(log_factors))**2))
                model_fits[name] = {
                    'parameters': popt,
                    'covariance': pcov,
                    'r_squared': r_squared
                }
            except:
                continue

        reduction_data['model_fits'] = model_fits
        return reduction_data

    def analyze_bottleneck_structure(self, results):
        G = nx.DiGraph()

        for dim in range(2, 8):
            for prop in ['R', 'S', 'T']:
                node = f"{prop}_{dim}"
                predicted = getattr(results[dim-2], f"{prop}_bound")
                actual = self.base.known_categories[dim].known_coherences if dim in self.base.known_categories else None

                G.add_node(node, dimension=dim, property=prop, predicted=predicted, actual=actual)

                if dim > 2:
                    prev_node = f"{prop}_{dim-1}"
                    G.add_edge(prev_node, node, type='dimensional')

                if prop == 'R':
                    G.add_edge(node, f"S_{dim}", type='interaction')
                if prop == 'S':
                    G.add_edge(node, f"T_{dim}", type='interaction')

        # Handle disconnected components for path length calculations
        components = list(nx.strongly_connected_components(G))
        largest_component = max(components, key=len)
        largest_subgraph = G.subgraph(largest_component)

        analysis = {
            'centrality': {
                'betweenness': nx.betweenness_centrality(G),
                'eigenvector': nx.eigenvector_centrality_numpy(G),
                'katz': nx.katz_centrality_numpy(G)
            },
            'components': {
                'strongly_connected': components,
                'weakly_connected': list(nx.weakly_connected_components(G)),
                'largest_component_size': len(largest_component)
            },
            'paths': {
                'average_path_length': nx.average_shortest_path_length(largest_subgraph) if len(largest_component) > 1 else 0,
                'diameter': nx.diameter(largest_subgraph) if len(largest_component) > 1 else 0
            },
            'bottlenecks': self.identify_critical_bottlenecks(G)
        }

        return analysis

    def identify_critical_bottlenecks(self, G):
        bottlenecks = {
            'structural': list(nx.articulation_points(G.to_undirected())),
            'dimensional': [],
            'property_based': []
        }

        for node in G.nodes():
            dim = G.nodes[node]['dimension']
            successors = list(G.successors(node))
            cross_dim_impact = sum(1 for s in successors if G.nodes[s]['dimension'] > dim)
            if cross_dim_impact >= 2:
                bottlenecks['dimensional'].append(node)

            if G.nodes[node]['predicted'] > 0:
                actual = G.nodes[node]['actual']
                if actual and actual < G.nodes[node]['predicted'] / 2:
                    bottlenecks['property_based'].append(node)

        return bottlenecks

    def visualize_reduction_patterns(self, reduction_data):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(range(2, 2+len(reduction_data['factors'])), reduction_data['factors'], 'bo-')
        plt.yscale('log')
        plt.xlabel('Dimension')
        plt.ylabel('Reduction Factor')
        plt.title('Coherence Reduction Factors')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        x = np.arange(2, 8)
        plt.plot(x, reduction_data['log_factors'], 'ko', label='Data')
        for name, fit in reduction_data['model_fits'].items():
            if name == 'exponential':
                a, b = fit['parameters']
                y = reduction_data['models']['exponential'](x, a, b)
                plt.plot(x, y, '--', label=f'{name} (R²={fit["r_squared"]:.3f})')
        plt.xlabel('Dimension')
        plt.ylabel('Log Reduction Factor')
        plt.title('Reduction Model Fits')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        dims = list(reduction_data['cancellation_patterns'].keys())
        efficiencies = [data['efficiency'] for data in reduction_data['cancellation_patterns'].values()]
        plt.plot(dims, efficiencies, 'go-')
        plt.xlabel('Dimension')
        plt.ylabel('Cancellation Efficiency')
        plt.title('Coherence Cancellation Efficiency')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_bottleneck_structure(self, bottleneck_analysis):
        G = nx.DiGraph()

        for comp in bottleneck_analysis['components']['strongly_connected']:
            for node in comp:
                G.add_node(node)

        for node, cent in bottleneck_analysis['centrality']['betweenness'].items():
            G.nodes[node]['centrality'] = cent

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        node_sizes = [1000 * G.nodes[node].get('centrality', 0.1) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.title('Bottleneck Structure Visualization')
        plt.axis('off')
        plt.show()

    def generate_detailed_report(self, reduction_data, bottleneck_analysis):
        print("\nDETAILED COHERENCE ANALYSIS REPORT")
        print("=" * 50)

        print("\n1. Reduction Mechanism Analysis:")
        print("-" * 30)

        best_model = max(reduction_data['model_fits'].items(), key=lambda x: x[1]['r_squared'])
        print(f"Best fitting reduction model: {best_model[0]}")
        print(f"R-squared: {best_model[1]['r_squared']:.4f}")

        print("\nCancellation Patterns:")
        for dim, data in reduction_data['cancellation_patterns'].items():
            print(f"\nDimension {dim}:")
            print(f"  Theoretical max: {data['theoretical_max']}")
            print(f"  Actual: {data['actual']}")
            print(f"  Reduction ratio: {data['reduction_ratio']:.2f}")
            print(f"  Efficiency: {data['efficiency']:.2f}")

        print("\n2. Bottleneck Structure Analysis:")
        print("-" * 30)

        print("\nCritical Bottlenecks:")
        for btype, nodes in bottleneck_analysis['bottlenecks'].items():
            print(f"\n{btype.capitalize()}:")
            for node in nodes:
                print(f"  {node}")

        print("\nNetwork Metrics:")
        print(f"Average path length: {bottleneck_analysis['paths']['average_path_length']:.2f}")
        print(f"Network diameter: {bottleneck_analysis['paths']['diameter']}")

class HigherCategoryAnalyzer:
    def __init__(self):
        self.known_categories = {
            2: CategoryData(2, 1, 1, ["Categories", "Groupoids"]),
            3: CategoryData(3, 2, 4, ["Bicategories", "Braided monoidal categories"]),
            4: CategoryData(4, 5, 16, ["Tricategories", "Symmetric monoidal categories"]),
            5: CategoryData(5, 7, 64, ["Tetracategories"]),
            6: CategoryData(6, 11, 256, ["Pentacategories"]),
            7: CategoryData(7, 13, 1024, ["Theoretical constructions only"])
        }
        self.base_rates = (0.086160, 0.765047, 0.766237)

    def theoretical_coherence_bound(self, dim):
        return int(factorial(dim-1))

    def measure_reflexivity(self, P_xx, dim):
        return 1 - (np.linalg.norm(P_xx - np.eye(dim), 'fro') / np.linalg.norm(np.eye(dim), 'fro'))

    def measure_symmetry(self, P_xy, P_yx, dim):
        return 1 - (np.linalg.norm(P_xy - P_yx.T, 'fro') / (np.linalg.norm(P_xy, 'fro') + np.linalg.norm(P_yx, 'fro')))

    def measure_transitivity(self, P_xy, P_yz, P_xz, dim):
        composition = P_xy @ P_yz
        return 1 - (np.linalg.norm(composition - P_xz, 'fro') / (np.linalg.norm(composition, 'fro') + np.linalg.norm(P_xz, 'fro')))

    def measure_composite_property(self, prop1, prop2):
        return np.sqrt(prop1 * prop2)

    def measure_higher_order(self, R, S, T):
        return (R * S * T) ** (1/3)

    def measure_extended_properties(self, dim, samples=1000):
        results = {
            'basic': {'R': [], 'S': [], 'T': []},
            'composite': {'RS': [], 'ST': [], 'RT': []},
            'higher': {'RST': []}
        }

        for _ in range(samples):
            points = [np.random.uniform(-1, 1, dim) for _ in range(4)]
            paths = {}

            paths[(0,0)] = create_path_space(points[0], points[0], dim)

            for i, j in itertools.combinations(range(4), 2):
                paths[(i,j)] = create_path_space(points[i], points[j], dim)
                paths[(j,i)] = create_path_space(points[j], points[i], dim)

            R = self.measure_reflexivity(paths[(0,0)], dim)
            S = self.measure_symmetry(paths[(0,1)], paths[(1,0)], dim)
            T = self.measure_transitivity(paths[(0,1)], paths[(1,2)], paths[(0,2)], dim)

            RS = self.measure_composite_property(R, S)
            ST = self.measure_composite_property(S, T)
            RT = self.measure_composite_property(R, T)

            RST = self.measure_higher_order(R, S, T)

            for key, value in zip(['R', 'S', 'T'], [R, S, T]):
                results['basic'][key].append(value)
            for key, value in zip(['RS', 'ST', 'RT'], [RS, ST, RT]):
                results['composite'][key].append(value)
            results['higher']['RST'].append(RST)

        return {k1: {k2: np.mean(v2) for k2, v2 in v1.items()}
                for k1, v1 in results.items()}

    def calculate_coherence_bound(self, property_value, dim):
        return int(np.ceil(-np.log(1 - property_value) / self.base_rates[0] * dim))

    def predict_total_coherences(self, R_bound, S_bound, T_bound, dim):
        weights = [1, dim/(dim+1), dim/(dim+2)]
        return int(np.ceil(np.average([R_bound, S_bound, T_bound], weights=weights)))

    def calculate_stability_metric(self, properties):
        basic_vals = list(properties['basic'].values())
        composite_vals = list(properties['composite'].values())
        higher_vals = list(properties['higher'].values())
        return (0.5 * np.mean(basic_vals) + 0.3 * np.mean(composite_vals) + 0.2 * np.mean(higher_vals))

    def analyze_coherence_structure(self, max_dim=10):
        results = []

        for dim in range(2, max_dim + 1):
            properties = self.measure_extended_properties(dim)

            R_bound = self.calculate_coherence_bound(properties['basic']['R'], dim)
            S_bound = self.calculate_coherence_bound(properties['basic']['S'], dim)
            T_bound = self.calculate_coherence_bound(properties['basic']['T'], dim)

            predicted = self.predict_total_coherences(R_bound, S_bound, T_bound, dim)

            actual = self.known_categories[dim].known_coherences if dim in self.known_categories else None

            stability = self.calculate_stability_metric(properties)

            results.append(CoherenceResult(
                dimension=dim,
                R_bound=R_bound,
                S_bound=S_bound,
                T_bound=T_bound,
                predicted_coherences=predicted,
                actual_coherences=actual,
                stability_metric=stability
            ))

        return results

    def analyze_key_findings(self, results):
        print("\nKey Findings:")
        stability_threshold = 0.9
        transition_dim = next((r.dimension for r in results if r.stability_metric < stability_threshold), None)
        print(f"Stability transition occurs at dimension: {transition_dim}")

        accuracy = []
        for r in results:
            if r.actual_coherences is not None:
                rel_error = abs(r.predicted_coherences - r.actual_coherences) / r.actual_coherences
                accuracy.append(1 - rel_error)
        print(f"Average prediction accuracy: {np.mean(accuracy):.4f}")

    def analyze_theoretical_implications(self, results):
        print("\nTheoretical Implications:")
        growth_rates = []
        for i in range(1, len(results)):
            if results[i].actual_coherences and results[i-1].actual_coherences:
                growth = results[i].actual_coherences / results[i-1].actual_coherences
                growth_rates.append(growth)

        print(f"Average growth rate: {np.mean(growth_rates):.2f}")
        print(f"Growth pattern: {'Super-exponential' if np.mean(growth_rates) > np.e else 'Sub-exponential'}")

    def generate_recommendations(self, results):
        print("\nPractical Recommendations:")
        optimal_dim = max((r.dimension for r in results if r.stability_metric > 0.9), default=None)
        print(f"Optimal working dimension: {optimal_dim}")

        high_bound_dims = [r.dimension for r in results if r.R_bound > 20]
        if high_bound_dims:
            print(f"Consider coherence reduction strategies for dimensions: {high_bound_dims}")

    def analyze_obstruction_theory(self):
        G = nx.DiGraph()

        for dim in range(2, 8):
            for prop in ['R', 'S', 'T']:
                node = f"{prop}_{dim}"
                G.add_node(node)

                if dim > 2:
                    prev_node = f"{prop}_{dim-1}"
                    G.add_edge(prev_node, node)

        cycles = list(nx.simple_cycles(G))
        components = list(nx.strongly_connected_components(G))

        return {
            'cycles': cycles,
            'components': components,
            'bottlenecks': self.find_bottlenecks(G)
        }

    def find_bottlenecks(self, G):
        centrality = nx.betweenness_centrality(G)
        threshold = np.mean(list(centrality.values())) + np.std(list(centrality.values()))
        return [node for node, cent in centrality.items() if cent > threshold]

    def analyze_obstructions(self, obstruction_analysis):
        print("\nObstruction Analysis:")
        print(f"Number of cycles: {len(obstruction_analysis['cycles'])}")
        print(f"Number of strongly connected components: {len(obstruction_analysis['components'])}")
        print(f"Bottleneck nodes: {obstruction_analysis['bottlenecks']}")

    def generate_report(self, results):
        print("\nCOMPREHENSIVE COHERENCE ANALYSIS REPORT")
        print("=" * 50)

        print("\n1. Dimension-wise Analysis:")
        for r in results:
            print(f"\nDimension {r.dimension}:")
            print(f"  R-bound: {r.R_bound}")
            print(f"  S-bound: {r.S_bound}")
            print(f"  T-bound: {r.T_bound}")
            print(f"  Predicted total: {r.predicted_coherences}")
            if r.actual_coherences:
                print(f"  Actual total: {r.actual_coherences}")
            print(f"  Stability: {r.stability_metric:.4f}")

        print("\n2. Key Findings:")
        self.analyze_key_findings(results)

        print("\n3. Theoretical Implications:")
        self.analyze_theoretical_implications(results)

        print("\n4. Practical Recommendations:")
        self.generate_recommendations(results)

    def run_full_analysis(self):
        results = self.analyze_coherence_structure()

        reduction_analyzer = CoherenceReductionAnalyzer(self)
        reduction_data = reduction_analyzer.analyze_reduction_mechanism(results)
        bottleneck_analysis = reduction_analyzer.analyze_bottleneck_structure(results)

        reduction_analyzer.visualize_reduction_patterns(reduction_data)
        reduction_analyzer.visualize_bottleneck_structure(bottleneck_analysis)
        reduction_analyzer.generate_detailed_report(reduction_data, bottleneck_analysis)

        self.generate_report(results)
        obstruction_analysis = self.analyze_obstruction_theory()
        self.analyze_obstructions(obstruction_analysis)

if __name__ == "__main__":
    np.random.seed(42)
    analyzer = HigherCategoryAnalyzer()
    analyzer.run_full_analysis()
