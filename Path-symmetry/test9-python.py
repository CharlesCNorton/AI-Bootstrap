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

# First, define our data classes
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

# Define utility functions
def create_path_space(x, y, dim):
    """Create path space matrix with controlled perturbation"""
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * dim)
    perturbation = np.random.uniform(-1, 1, (dim, dim))
    return np.eye(dim) + epsilon * np.exp(-0.3 * dist) * perturbation

def exp_decay(x, a, b, c):
    """Exponential decay function with parameters"""
    return a * np.exp(-b * x) + c

# Main analyzer class
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

    def sylvester_sequence(self, n: int) -> List[int]:
        """Generate Sylvester's sequence for coherence growth estimation"""
        sequence = [2]
        while len(sequence) < n:
            sequence.append(np.prod(sequence) + 1)
        return sequence

    def theoretical_coherence_bound(self, dim: int) -> int:
        """Calculate theoretical upper bound on coherences"""
        return int(factorial(dim-1))

    def stasheff_associahedra(self, dim: int) -> int:
        """Calculate vertices of Stasheff associahedra"""
        if dim <= 1:
            return 1
        return sum(self.stasheff_associahedra(k) * self.stasheff_associahedra(dim-k)
                  for k in range(1, dim))

    def measure_reflexivity(self, P_xx: np.ndarray, dim: int) -> float:
        """Measure reflexivity property"""
        return 1 - (np.linalg.norm(P_xx - np.eye(dim), 'fro') /
                   np.linalg.norm(np.eye(dim), 'fro'))

    def measure_symmetry(self, P_xy: np.ndarray, P_yx: np.ndarray, dim: int) -> float:
        """Measure symmetry property"""
        return 1 - (np.linalg.norm(P_xy - P_yx.T, 'fro') /
                   (np.linalg.norm(P_xy, 'fro') + np.linalg.norm(P_yx, 'fro')))

    def measure_transitivity(self, P_xy: np.ndarray, P_yz: np.ndarray,
                           P_xz: np.ndarray, dim: int) -> float:
        """Measure transitivity property"""
        composition = P_xy @ P_yz
        return 1 - (np.linalg.norm(composition - P_xz, 'fro') /
                   (np.linalg.norm(composition, 'fro') + np.linalg.norm(P_xz, 'fro')))

    def measure_composite_property(self, prop1: float, prop2: float) -> float:
        """Measure interaction between two properties"""
        return np.sqrt(prop1 * prop2)

    def measure_higher_order(self, R: float, S: float, T: float) -> float:
        """Measure higher-order interaction between all properties"""
        return (R * S * T) ** (1/3)

    def measure_extended_properties(self, dim: int, samples: int = 1000) -> Dict:
        """Enhanced property measurement including higher-order effects"""
        results = {
            'basic': {'R': [], 'S': [], 'T': []},
            'composite': {'RS': [], 'ST': [], 'RT': []},
            'higher': {'RST': []}
        }

        for _ in range(samples):
            points = [np.random.uniform(-1, 1, dim) for _ in range(4)]
            paths = {}

            # Include reflexive paths explicitly
            paths[(0,0)] = create_path_space(points[0], points[0], dim)

            # Generate all other paths
            for i, j in itertools.combinations(range(4), 2):
                paths[(i,j)] = create_path_space(points[i], points[j], dim)
                paths[(j,i)] = create_path_space(points[j], points[i], dim)  # Include reverse paths

            # Basic properties
            R = self.measure_reflexivity(paths[(0,0)], dim)
            S = self.measure_symmetry(paths[(0,1)], paths[(1,0)], dim)
            T = self.measure_transitivity(paths[(0,1)], paths[(1,2)], paths[(0,2)], dim)

            # Composite properties
            RS = self.measure_composite_property(R, S)
            ST = self.measure_composite_property(S, T)
            RT = self.measure_composite_property(R, T)

            # Higher-order interaction
            RST = self.measure_higher_order(R, S, T)

            # Store results
            for key, value in zip(['R', 'S', 'T'], [R, S, T]):
                results['basic'][key].append(value)
            for key, value in zip(['RS', 'ST', 'RT'], [RS, ST, RT]):
                results['composite'][key].append(value)
            results['higher']['RST'].append(RST)

        return {k1: {k2: np.mean(v2) for k2, v2 in v1.items()}
                for k1, v1 in results.items()}

    def calculate_coherence_bound(self, property_value: float, dim: int) -> int:
        """Calculate coherence bound based on property value"""
        return int(np.ceil(-np.log(1 - property_value) / self.base_rates[0] * dim))

    def predict_total_coherences(self, R_bound: int, S_bound: int,
                               T_bound: int, dim: int) -> int:
        """Predict total number of coherence conditions"""
        weights = [1, dim/(dim+1), dim/(dim+2)]
        return int(np.ceil(np.average([R_bound, S_bound, T_bound], weights=weights)))

    def calculate_stability_metric(self, properties: Dict) -> float:
        """Calculate stability metric for coherence structure"""
        basic_vals = list(properties['basic'].values())
        composite_vals = list(properties['composite'].values())
        higher_vals = list(properties['higher'].values())

        return (0.5 * np.mean(basic_vals) +
                0.3 * np.mean(composite_vals) +
                0.2 * np.mean(higher_vals))

    def analyze_coherence_structure(self, max_dim: int = 10) -> List[CoherenceResult]:
        """Comprehensive analysis of coherence structures"""
        results = []

        for dim in range(2, max_dim + 1):
            properties = self.measure_extended_properties(dim)

            R_bound = self.calculate_coherence_bound(properties['basic']['R'], dim)
            S_bound = self.calculate_coherence_bound(properties['basic']['S'], dim)
            T_bound = self.calculate_coherence_bound(properties['basic']['T'], dim)

            predicted = self.predict_total_coherences(R_bound, S_bound, T_bound, dim)

            actual = (self.known_categories[dim].known_coherences
                     if dim in self.known_categories else None)

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

    def find_bottlenecks(self, G: nx.DiGraph) -> List[str]:
        """Find bottleneck nodes in coherence graph"""
        centrality = nx.betweenness_centrality(G)
        threshold = np.mean(list(centrality.values())) + np.std(list(centrality.values()))
        return [node for node, cent in centrality.items() if cent > threshold]

    def analyze_obstruction_theory(self):
        """Analyze obstruction patterns in higher coherences"""
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

    def visualize_results(self, results: List[CoherenceResult]):
        """Create comprehensive visualization of results"""
        plt.figure(figsize=(20, 15))

        # Plot 1: Coherence bounds
        plt.subplot(2, 2, 1)
        dims = [r.dimension for r in results]
        r_bounds = [r.R_bound for r in results]
        s_bounds = [r.S_bound for r in results]
        t_bounds = [r.T_bound for r in results]

        plt.plot(dims, r_bounds, 'b-o', label='R bound')
        plt.plot(dims, s_bounds, 'r-o', label='S bound')
        plt.plot(dims, t_bounds, 'g-o', label='T bound')
        plt.yscale('log')
        plt.xlabel('Dimension')
        plt.ylabel('Coherence Bound')
        plt.title('Coherence Bounds vs Dimension')
        plt.legend()
        plt.grid(True)

        # Plot 2: Predicted vs Actual
        plt.subplot(2, 2, 2)
        predicted = [r.predicted_coherences for r in results]
        actual = [r.actual_coherences if r.actual_coherences is not None else np.nan
                 for r in results]

        plt.plot(dims, predicted, 'b-o', label='Predicted')
        plt.plot(dims, actual, 'r-o', label='Actual')
        plt.yscale('log')
        plt.xlabel('Dimension')
        plt.ylabel('Total Coherences')
        plt.title('Predicted vs Actual Coherences')
        plt.legend()
        plt.grid(True)

        # Plot 3: Stability Metric
        plt.subplot(2, 2, 3)
        stability = [r.stability_metric for r in results]
        plt.plot(dims, stability, 'k-o')
        plt.xlabel('Dimension')
        plt.ylabel('Stability Metric')
        plt.title('Coherence Stability vs Dimension')
        plt.grid(True)

        # Plot 4: Theoretical Bounds
        plt.subplot(2, 2, 4)
        theoretical = [self.theoretical_coherence_bound(d) for d in dims]
        stasheff = [self.stasheff_associahedra(d) for d in dims]
        sylvester = self.sylvester_sequence(len(dims))

        plt.plot(dims, theoretical, 'b-o', label='Factorial Bound')
        plt.plot(dims, stasheff, 'r-o', label='Stasheff Numbers')
        plt.plot(dims, sylvester[:len(dims)], 'g-o', label='Sylvester Sequence')
        plt.yscale('log')
        plt.xlabel('Dimension')
        plt.ylabel('Bound Value')
        plt.title('Theoretical Bounds Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def analyze_key_findings(self, results: List[CoherenceResult]):
        """Analyze and print key findings"""
        print("\nKey Findings:")
        stability_threshold = 0.9
        transition_dim = next((r.dimension for r in results
                             if r.stability_metric < stability_threshold), None)
        print(f"Stability transition occurs at dimension: {transition_dim}")

        accuracy = []
        for r in results:
            if r.actual_coherences is not None:
                rel_error = abs(r.predicted_coherences - r.actual_coherences) / r.actual_coherences
                accuracy.append(1 - rel_error)
        print(f"Average prediction accuracy: {np.mean(accuracy):.4f}")

    def analyze_theoretical_implications(self, results: List[CoherenceResult]):
        """Analyze theoretical implications of results"""
        print("\nTheoretical Implications:")
        growth_rates = []
        for i in range(1, len(results)):
            if results[i].actual_coherences and results[i-1].actual_coherences:
                growth = results[i].actual_coherences / results[i-1].actual_coherences
                growth_rates.append(growth)

        print(f"Average growth rate: {np.mean(growth_rates):.2f}")
        print(f"Growth pattern: {'Super-exponential' if np.mean(growth_rates) > np.e else 'Sub-exponential'}")

    def generate_recommendations(self, results: List[CoherenceResult]):
        """Generate practical recommendations"""
        print("\nPractical Recommendations:")
        optimal_dim = max((r.dimension for r in results
                          if r.stability_metric > 0.9), default=None)
        print(f"Optimal working dimension: {optimal_dim}")

        high_bound_dims = [r.dimension for r in results if r.R_bound > 20]
        if high_bound_dims:
            print(f"Consider coherence reduction strategies for dimensions: {high_bound_dims}")

    def analyze_obstructions(self, obstruction_analysis: Dict):
        """Analyze obstruction patterns"""
        print("\nObstruction Analysis:")
        print(f"Number of cycles: {len(obstruction_analysis['cycles'])}")
        print(f"Number of strongly connected components: {len(obstruction_analysis['components'])}")
        print(f"Bottleneck nodes: {obstruction_analysis['bottlenecks']}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        results = self.analyze_coherence_structure()
        self.visualize_results(results)
        self.generate_report(results)

        obstruction_analysis = self.analyze_obstruction_theory()
        self.analyze_obstructions(obstruction_analysis)

    def generate_report(self, results: List[CoherenceResult]):
        """Generate comprehensive analysis report"""
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

if __name__ == "__main__":
    np.random.seed(42)
    analyzer = HigherCategoryAnalyzer()
    analyzer.run_full_analysis()
