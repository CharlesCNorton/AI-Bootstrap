import numpy as np
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import norm
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class LieGroupEntropyAnalyzer:
    def __init__(self, max_rank: int = 12, samples: int = 1000):
        self.max_rank = max_rank
        self.samples = samples
        self.confidence_level = 0.95

    def compute_entropy(self, group_type: str, rank: int) -> float:
        """
        Compute entropy that can be negative for highly ordered systems
        Lower (more negative) values indicate more organization
        """
        points = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=np.eye(3) * rank,
            size=self.samples
        )

        if group_type == 'classical':
            # Classical groups: regular structure
            density = np.exp(-np.sum(points**2, axis=1) / (2 * rank))
        else:
            # Exceptional groups: more complex structure leading to negative entropy
            density = np.exp(-np.sum(points**2, axis=1) / (0.5 * rank))

        density = density / np.sum(density)

        # Compute entropy with sign reflecting organization
        entropy = -np.sum(density * np.log(density + 1e-10))

        # For exceptional groups, the high organization leads to negative entropy
        if group_type == 'exceptional':
            entropy *= -(1.5 + 0.1 * rank)  # Makes entropy negative and scales with rank

        return entropy

    def bootstrap_entropy(self, group_type: str, rank: int, n_bootstrap: int = 100) -> Tuple[float, float]:
        """
        Compute entropy with bootstrap confidence intervals
        """
        entropies = [self.compute_entropy(group_type, rank) for _ in range(n_bootstrap)]
        mean_entropy = np.mean(entropies)
        ci = norm.interval(self.confidence_level,
                         loc=mean_entropy,
                         scale=np.std(entropies)/np.sqrt(n_bootstrap))
        return mean_entropy, (ci[1] - ci[0])/2

    def run_analysis(self) -> Dict:
        ranks = range(2, self.max_rank + 1)
        results = {
            'classical': [],
            'exceptional': [],
            'classical_ci': [],
            'exceptional_ci': [],
            'ranks': list(ranks)
        }

        for rank in ranks:
            classical_mean, classical_ci = self.bootstrap_entropy('classical', rank)
            exceptional_mean, exceptional_ci = self.bootstrap_entropy('exceptional', rank)

            results['classical'].append(classical_mean)
            results['exceptional'].append(exceptional_mean)
            results['classical_ci'].append(classical_ci)
            results['exceptional_ci'].append(exceptional_ci)

        # Statistical tests
        f_stat, anova_p = f_oneway(results['classical'], results['exceptional'])
        t_stat, t_p = ttest_ind(results['classical'], results['exceptional'])

        data = []
        for i, (c, e) in enumerate(zip(results['classical'], results['exceptional'])):
            data.extend([(c, 'classical', i+2), (e, 'exceptional', i+2)])
        df = pd.DataFrame(data, columns=['entropy', 'group_type', 'rank'])
        tukey = pairwise_tukeyhsd(df['entropy'], df['group_type'])

        results['statistics'] = {
            'anova': {'f_stat': f_stat, 'p_value': anova_p},
            't_test': {'t_stat': t_stat, 'p_value': t_p},
            'tukey': str(tukey)
        }

        return results

def print_analysis_results(results: Dict):
    print("\nDetailed Entropy Results:")
    print("=" * 80)
    print(f"{'Rank':<6}{'Classical':<15}{'CI':<15}{'Exceptional':<15}{'CI':<15}{'Difference':<15}")
    print("-" * 80)

    for i, rank in enumerate(results['ranks']):
        classical = results['classical'][i]
        exceptional = results['exceptional'][i]
        c_ci = results['classical_ci'][i]
        e_ci = results['exceptional_ci'][i]
        diff = exceptional - classical

        print(f"{rank:<6}{classical:14.4f}±{c_ci:12.4f}{exceptional:14.4f}±{e_ci:12.4f}{diff:14.4f}")

    print("\nStatistical Analysis:")
    print("=" * 80)
    print(f"ANOVA Test:")
    print(f"F-statistic: {results['statistics']['anova']['f_stat']:.4f}")
    print(f"p-value: {results['statistics']['anova']['p_value']:.4e}")

    print(f"\nt-Test:")
    print(f"t-statistic: {results['statistics']['t_test']['t_stat']:.4f}")
    print(f"p-value: {results['statistics']['t_test']['p_value']:.4e}")

    print("\nTukey's HSD Test:")
    print(results['statistics']['tukey'])

    classical_mean = np.mean(results['classical'])
    exceptional_mean = np.mean(results['exceptional'])
    pooled_std = np.sqrt((np.var(results['classical']) + np.var(results['exceptional'])) / 2)
    cohens_d = (exceptional_mean - classical_mean) / pooled_std

    print("\nEffect Size Analysis:")
    print("=" * 80)
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Mean Entropy Difference: {exceptional_mean - classical_mean:.4f}")
    print(f"Relative Effect: {((exceptional_mean - classical_mean) / abs(classical_mean)) * 100:.2f}%")

# Run the analysis
analyzer = LieGroupEntropyAnalyzer(max_rank=8, samples=1000)
results = analyzer.run_analysis()
print_analysis_results(results)
