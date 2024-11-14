import numpy as np
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StabilityConfig:
    min_level: int = 1
    max_level: int = 200
    num_samples: int = 1000
    threshold_range: Tuple[float, float] = (0.1, 5.0)
    num_thresholds: int = 50
    epsilon_range: Tuple[float, float] = (-1.0, 1.0)
    test_dimensions: List[str] = None

    def __post_init__(self):
        if self.test_dimensions is None:
            self.test_dimensions = ['standard', 'cohomological', 'oscillatory', 'adaptive']

class ExtendedHomotopyTest:
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.thresholds = np.linspace(
            config.threshold_range[0],
            config.threshold_range[1],
            config.num_thresholds
        )

    def adaptive_scaling(self, n: int, mode: str = 'standard') -> float:
        if mode == 'standard':
            return 1 / (1 + n)
        elif mode == 'logarithmic':
            return 1 / (1 + np.log(n + 1))
        elif mode == 'exponential':
            return np.exp(-n/50)
        return 1 / (1 + n)

    def cup_product(self, x: float, y: float, n: int, mode: str = 'standard') -> float:
        if mode == 'standard':
            return np.sin(n * x) * np.cos(n * y)
        elif mode == 'advanced':
            return np.sin(n * x) * np.cos(n * y) * np.exp(-n/100)
        return np.sin(n * x) * np.cos(n * y)

    def oscillatory_term(self, n: int, value: float, mode: str = 'standard') -> float:
        if mode == 'standard':
            return np.cos(n * value)
        elif mode == 'damped':
            return np.cos(n * value) * np.exp(-n/100)
        elif mode == 'enhanced':
            return np.cos(n * value) * (1 + np.sin(n * value/2))
        return np.cos(n * value)

    def loop_space(self, n: int, a0: float, epsilon: float,
                  scaling_mode: str = 'standard',
                  oscillatory_mode: str = 'standard') -> float:
        scaled_epsilon = epsilon * self.adaptive_scaling(n, scaling_mode)
        base_term = np.abs((a0 + scaled_epsilon) / 2)**(1/n)  # Using abs to avoid complex numbers
        osc_term = self.oscillatory_term(n, a0 + scaled_epsilon, oscillatory_mode)
        return base_term + osc_term

    def product_type(self, n: int, a01: float, a02: float, epsilon: float,
                    scaling_mode: str = 'standard',
                    oscillatory_mode: str = 'standard') -> float:
        scaled_epsilon = epsilon * self.adaptive_scaling(n, scaling_mode)
        term1 = (np.abs(a01 + scaled_epsilon)**(1/n) +
                 self.oscillatory_term(n, a01 + scaled_epsilon, oscillatory_mode))
        term2 = (np.abs(a02 - scaled_epsilon)**(1/n) +
                 self.oscillatory_term(n, a02 - scaled_epsilon, oscillatory_mode))
        return (term1 + term2) / 2

    def fibration_type(self, n: int, base: float, fiber1: float, fiber2: float,
                      epsilon: float, scaling_mode: str = 'standard',
                      oscillatory_mode: str = 'standard',
                      cohomology_mode: str = 'standard') -> float:
        scaled_epsilon = epsilon * self.adaptive_scaling(n, scaling_mode)

        base_term = (np.abs(base + scaled_epsilon)**(1/n) +
                    self.oscillatory_term(n, base, oscillatory_mode))

        fiber1_term = (np.abs(fiber1 + 0.5 * scaled_epsilon)**(1/(n+1)) +
                      self.oscillatory_term(n, fiber1, oscillatory_mode) +
                      self.cup_product(base, fiber1, n, cohomology_mode))

        fiber2_term = (np.abs(fiber2 + 0.25 * scaled_epsilon)**(1/(n+2)) +
                      self.oscillatory_term(n, fiber2, oscillatory_mode) +
                      self.cup_product(base, fiber2, n, cohomology_mode))

        return (base_term + fiber1_term/2 + fiber2_term/2) / 2

    def evaluate_stability(self, value: float, n: int,
                         scaling_mode: str = 'standard') -> Dict:
        return {threshold: abs(value) < (threshold * self.adaptive_scaling(n, scaling_mode))
                for threshold in self.thresholds}

    def process_level(self, n: int, dimension: str) -> Dict:
        stability_counts = {t: 0 for t in self.thresholds}
        value_distribution = []
        total = 0

        scaling_mode = 'logarithmic' if dimension == 'adaptive' else 'standard'
        oscillatory_mode = 'damped' if dimension == 'oscillatory' else 'standard'
        cohomology_mode = 'advanced' if dimension == 'cohomological' else 'standard'

        for _ in range(self.config.num_samples):
            a0 = np.random.uniform(-1, 1)
            a01, a02 = np.random.uniform(-1, 1, 2)
            base, fiber1, fiber2 = np.random.uniform(-1, 1, 3)
            epsilon = np.random.uniform(*self.config.epsilon_range)

            try:
                loop_val = self.loop_space(n, a0, epsilon, scaling_mode, oscillatory_mode)
                prod_val = self.product_type(n, a01, a02, epsilon, scaling_mode, oscillatory_mode)
                fib_val = self.fibration_type(n, base, fiber1, fiber2, epsilon,
                                            scaling_mode, oscillatory_mode, cohomology_mode)

                values = [loop_val, prod_val, fib_val]
                if all(np.isfinite(v) for v in values):
                    avg_val = np.mean(values)
                    value_distribution.append(avg_val)
                    stability = self.evaluate_stability(avg_val, n, scaling_mode)
                    for t in self.thresholds:
                        if stability[t]:
                            stability_counts[t] += 1
                    total += 1
            except:
                continue

        if total == 0:
            return None

        distribution_stats = {
            'mean': np.mean(value_distribution),
            'std': np.std(value_distribution),
            'skew': stats.skew(value_distribution),
            'kurtosis': stats.kurtosis(value_distribution)
        }

        return {
            'stability_ratios': {t: count/total for t, count in stability_counts.items()},
            'distribution': distribution_stats,
            'total_samples': total
        }

    def run_test(self) -> Dict:
        results = {dim: {} for dim in self.config.test_dimensions}
        levels = list(range(self.config.min_level, self.config.max_level + 1))

        total_iterations = len(levels) * len(self.config.test_dimensions)
        current_iteration = 0

        for dimension in self.config.test_dimensions:
            for n in levels:
                result = self.process_level(n, dimension)
                if result is not None:
                    results[dimension][n] = result

                # Progress update
                current_iteration += 1
                if current_iteration % 10 == 0:
                    progress = (current_iteration / total_iterations) * 100
                    print(f"Progress: {progress:.1f}%")

        return results

def analyze_and_plot_results(results: Dict, config: StabilityConfig):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Stability Analysis Across Different Dimensions')

    # Plot 1: Stability Decay
    ax = axes[0, 0]
    for dim in results:
        levels = sorted(results[dim].keys())
        stability = [results[dim][n]['stability_ratios'][config.threshold_range[1]]
                    for n in levels]
        ax.plot(levels, stability, label=dim)
    ax.set_xlabel('Homotopy Level')
    ax.set_ylabel('Stability Ratio')
    ax.set_title('Stability Decay by Dimension')
    ax.legend()

    # Plot 2: Distribution Statistics
    ax = axes[0, 1]
    for dim in results:
        levels = sorted(results[dim].keys())
        means = [results[dim][n]['distribution']['mean'] for n in levels]
        stds = [results[dim][n]['distribution']['std'] for n in levels]
        ax.errorbar(levels, means, yerr=stds, label=dim, alpha=0.5)
    ax.set_xlabel('Homotopy Level')
    ax.set_ylabel('Mean Value')
    ax.set_title('Distribution Statistics')
    ax.legend()

    # Plot 3: Threshold Sensitivity
    ax = axes[1, 0]
    selected_levels = [1, 10, 50, 100]
    for level in selected_levels:
        for dim in results:
            if level in results[dim]:
                thresholds = sorted(results[dim][level]['stability_ratios'].keys())
                ratios = [results[dim][level]['stability_ratios'][t] for t in thresholds]
                ax.plot(thresholds, ratios,
                       label=f'{dim} (n={level})',
                       alpha=0.7)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Stability Ratio')
    ax.set_title('Threshold Sensitivity Analysis')
    ax.legend()

    # Plot 4: Statistical Moments
    ax = axes[1, 1]
    for dim in results:
        levels = sorted(results[dim].keys())
        skew = [results[dim][n]['distribution']['skew'] for n in levels]
        kurt = [results[dim][n]['distribution']['kurtosis'] for n in levels]
        ax.scatter(skew, kurt, label=dim, alpha=0.5)
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Distribution Moments')
    ax.legend()

    plt.tight_layout()
    plt.savefig('stability_analysis.png')
    plt.close()

def print_summary_statistics(results: Dict):
    print("\nSummary Statistics:")

    # Find closest threshold to 3.0
    def find_nearest_threshold(thresholds, value):
        return min(thresholds, key=lambda x: abs(x - value))

    for dimension in results:
        print(f"\n{dimension.upper()} DIMENSION:")
        levels = sorted(results[dimension].keys())

        # Get available thresholds from the first level's data
        thresholds = sorted(results[dimension][levels[0]]['stability_ratios'].keys())
        target_threshold = find_nearest_threshold(thresholds, 3.0)

        # Low levels (1-5)
        low_levels = [n for n in levels if n <= 5]
        if low_levels:
            print("\nLow Levels (1-5):")
            for n in low_levels:
                stats = results[dimension][n]['distribution']
                stability = results[dimension][n]['stability_ratios'][target_threshold]
                print(f"Level {n}: Stability={stability:.4f}, Mean={stats['mean']:.4f}, "
                      f"Std={stats['std']:.4f}, Threshold={target_threshold:.4f}")

        # Mid levels (20-50)
        mid_levels = [n for n in levels if 20 <= n <= 50]
        if mid_levels:
            print("\nMid Levels (20-50):")
            for n in [20, 35, 50]:
                if n in results[dimension]:
                    stats = results[dimension][n]['distribution']
                    stability = results[dimension][n]['stability_ratios'][target_threshold]
                    print(f"Level {n}: Stability={stability:.4f}, Mean={stats['mean']:.4f}, "
                          f"Std={stats['std']:.4f}, Threshold={target_threshold:.4f}")

        # High levels (100+)
        high_levels = [n for n in levels if n >= 100]
        if high_levels:
            print("\nHigh Levels (100+):")
            for n in [100, 150, 200]:
                if n in results[dimension]:
                    stats = results[dimension][n]['distribution']
                    stability = results[dimension][n]['stability_ratios'][target_threshold]
                    print(f"Level {n}: Stability={stability:.4f}, Mean={stats['mean']:.4f}, "
                          f"Std={stats['std']:.4f}, Threshold={target_threshold:.4f}")
                    print(f"Level {n}: Stability={stability:.4f}, Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

if __name__ == "__main__":
    config = StabilityConfig(
        min_level=1,
        max_level=200,
        num_samples=1000,
        threshold_range=(0.1, 5.0),
        num_thresholds=50,
        epsilon_range=(-1.0, 1.0)
    )

    start_time = time.time()
    test = ExtendedHomotopyTest(config)
    results = test.run_test()

    print(f"\nTest completed in {time.time() - start_time:.2f} seconds")

    print_summary_statistics(results)
    analyze_and_plot_results(results, config)
