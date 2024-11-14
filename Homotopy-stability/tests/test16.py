import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class HomotopyStabilityTest:
    def __init__(self, max_level: int = 5, num_trials: int = 1000, epsilon_range: float = 0.5):
        self.max_level = max_level
        self.num_trials = num_trials
        self.epsilon_range = epsilon_range

    def cup_product(self, x: float, y: float, n: int) -> float:
        """Compute cohomological cup product interaction"""
        return np.sin(n * x) * np.cos(n * y)

    def adaptive_epsilon(self, n: int, base_epsilon: float) -> float:
        """Scale epsilon based on homotopy level"""
        return base_epsilon / (1 + n)

    def loop_space(self, a0: float, epsilon: float, n: int) -> float:
        """
        Evaluate loop space with averaging and oscillatory damping
        H_n: Loop space at level n
        """
        scaled_epsilon = self.adaptive_epsilon(n, epsilon)
        base_term = ((a0 + scaled_epsilon) / 2) ** (1/n)
        oscillatory_term = np.cos(n * (a0 + scaled_epsilon))
        damping = 1 / (1 + abs(scaled_epsilon))
        return base_term + damping * oscillatory_term

    def product_type(self, a01: float, a02: float, epsilon: float, n: int) -> float:
        """
        Evaluate product type with balanced perturbations
        P_n: Product type at level n
        """
        scaled_epsilon = self.adaptive_epsilon(n, epsilon)
        term1 = ((a01 + scaled_epsilon) ** (1/n) + np.cos(n * (a01 + scaled_epsilon)))
        term2 = ((a02 - scaled_epsilon) ** (1/n) + np.sin(n * (a02 - scaled_epsilon)))
        return (term1 + term2) / 2

    def fibration_type(self, base: float, fiber1: float, fiber2: float,
                      epsilon: float, n: int) -> float:
        """
        Evaluate fibration type with cohomological interactions
        F_n: Fibration type at level n
        """
        scaled_epsilon = self.adaptive_epsilon(n, epsilon)

        # Base space term
        base_term = (base + scaled_epsilon) ** (1/n) + np.cos(n * base)

        # Fiber terms with cup products
        fiber1_term = ((fiber1 + 0.5 * scaled_epsilon) ** (1/(n+1)) +
                      np.sin(n * fiber1) + self.cup_product(base, fiber1, n))

        fiber2_term = ((fiber2 + 0.25 * scaled_epsilon) ** (1/(n+2)) +
                      np.sin(n * fiber2) + self.cup_product(base, fiber2, n))

        return (base_term + fiber1_term/2 + fiber2_term/2) / 2

    def evaluate_stability(self) -> Dict:
        """
        Comprehensive stability evaluation across all homotopy types
        """
        results = {
            'loop_space': {'positive': [], 'negative': []},
            'product_type': {'positive': [], 'negative': []},
            'fibration_type': {'positive': [], 'negative': []}
        }

        for n in range(1, self.max_level + 1):
            for _ in range(self.num_trials):
                # Generate random initial conditions
                a0 = np.random.uniform(-1, 1)
                a01, a02 = np.random.uniform(-1, 1, 2)
                base, fiber1, fiber2 = np.random.uniform(-1, 1, 3)

                # Test both positive and negative perturbations
                for epsilon_sign in [1, -1]:
                    epsilon = np.random.uniform(0, self.epsilon_range) * epsilon_sign
                    key = 'positive' if epsilon_sign > 0 else 'negative'

                    try:
                        # Compute stability values
                        loop_val = self.loop_space(a0, epsilon, n)
                        prod_val = self.product_type(a01, a02, epsilon, n)
                        fib_val = self.fibration_type(base, fiber1, fiber2, epsilon, n)

                        # Store results if they're valid
                        if all(map(np.isfinite, [loop_val, prod_val, fib_val])):
                            results['loop_space'][key].append(loop_val)
                            results['product_type'][key].append(prod_val)
                            results['fibration_type'][key].append(fib_val)
                    except:
                        continue

        return results

    def analyze_results(self, results: Dict) -> Dict:
        """
        Statistical analysis of stability results
        """
        analysis = {}

        for htype in results:
            analysis[htype] = {}
            for ptype in ['positive', 'negative']:
                values = np.array(results[htype][ptype])
                values = values[np.isfinite(values)]

                # Define stability threshold
                stability_threshold = 2.0

                analysis[htype][ptype] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'stability_ratio': np.mean(np.abs(values) < stability_threshold),
                    'kurtosis': stats.kurtosis(values),
                    'skewness': stats.skew(values)
                }

        return analysis

def run_stability_test():
    """
    Execute complete stability test and return results
    """
    test = HomotopyStabilityTest(max_level=5, num_trials=1000)
    results = test.evaluate_stability()
    analysis = test.analyze_results(results)

    # Print detailed results
    for htype in analysis:
        print(f"\nResults for {htype}:")
        for ptype in ['positive', 'negative']:
            print(f"\n{ptype.capitalize()} perturbations:")
            for metric, value in analysis[htype][ptype].items():
                print(f"{metric}: {value:.4f}")

    return analysis

if __name__ == "__main__":
    analysis = run_stability_test()
