import numpy as np
import math
from typing import Dict, List, Tuple

class UnifiedHomotopyTest:
    def __init__(self, low_level=1, high_level=100, num_thresholds=20):
        self.low_level = low_level
        self.high_level = high_level
        self.thresholds = np.linspace(0.1, 3.0, num_thresholds)

    def adaptive_scaling(self, n: int) -> float:
        return 1 / (1 + n)

    def loop_space(self, n: int, a0: float, epsilon: float) -> float:
        scaled_epsilon = epsilon * self.adaptive_scaling(n)
        return ((a0 + scaled_epsilon) / 2)**(1/n) + math.cos(n * (a0 + scaled_epsilon))

    def evaluate_stability(self, value: float, n: int) -> Dict:
        return {threshold: abs(value) < (threshold * self.adaptive_scaling(n))
                for threshold in self.thresholds}

    def run_test(self, num_samples: int = 1000) -> Dict:
        results = {
            'low_level': {},  # n ∈ [1,5]
            'high_level': {}  # n ∈ [20,100]
        }

        # Test low levels (1-5)
        for n in range(1, 6):
            stability_counts = {t: 0 for t in self.thresholds}
            total = 0

            for _ in range(num_samples):
                a0 = np.random.uniform(-1, 1)
                epsilon = np.random.uniform(-0.5, 0.5)

                value = self.loop_space(n, a0, epsilon)
                if np.isfinite(value):
                    stability = self.evaluate_stability(value, n)
                    for t in self.thresholds:
                        if stability[t]:
                            stability_counts[t] += 1
                    total += 1

            results['low_level'][n] = {
                'stability_ratios': {t: count/total if total > 0 else 0
                                   for t, count in stability_counts.items()}
            }

        # Test high levels (20-100)
        for n in [20, 40, 60, 80, 100]:
            stability_counts = {t: 0 for t in self.thresholds}
            total = 0

            for _ in range(num_samples):
                a0 = np.random.uniform(-1, 1)
                epsilon = np.random.uniform(-0.5, 0.5)

                value = self.loop_space(n, a0, epsilon)
                if np.isfinite(value):
                    stability = self.evaluate_stability(value, n)
                    for t in self.thresholds:
                        if stability[t]:
                            stability_counts[t] += 1
                    total += 1

            results['high_level'][n] = {
                'stability_ratios': {t: count/total if total > 0 else 0
                                   for t, count in stability_counts.items()}
            }

        return results

# Run unified test
test = UnifiedHomotopyTest()
unified_results = test.run_test()

# Print results
print("\nLow Level Results (n ∈ [1,5]):")
for n, data in unified_results['low_level'].items():
    print(f"\nLevel {n}:")
    for threshold, ratio in data['stability_ratios'].items():
        if threshold in [0.1, 1.0, 2.0, 3.0]:  # Print select thresholds
            print(f"  Threshold {threshold:.1f}: {ratio:.4f}")

print("\nHigh Level Results (n ∈ [20,100]):")
for n, data in unified_results['high_level'].items():
    print(f"\nLevel {n}:")
    for threshold, ratio in data['stability_ratios'].items():
        if threshold in [0.1, 1.0, 2.0, 3.0]:  # Print select thresholds
            print(f"  Threshold {threshold:.1f}: {ratio:.4f}")
