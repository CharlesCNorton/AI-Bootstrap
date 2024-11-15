import numpy as np
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import bootstrap
import itertools
import logging

# Configure logging for detailed debug information
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def generate_data(n_samples, complexity_factor=1.0):
    """
    Generates synthetic data for various topological structures with
    added complexity and variability.
    """
    return [
        {
            "c_values": (np.random.uniform(1, 5, 5) * complexity_factor).tolist(),
            "x_values": (np.random.uniform(0.5, 2.5, 5) * complexity_factor).tolist(),
            "c_0": complexity_factor * np.random.uniform(0.8, 1.2)
        }
        for _ in range(n_samples)
    ]


# Function to calculate K_invariant with complexity adjustments
def calculate_k_invariant(c_values, x_values, c_0):
    linear_comb = sum(ci * xi for ci, xi in zip(c_values, x_values)) + c_0
    k_invariant = (linear_comb ** 2) + np.sin(linear_comb)
    return k_invariant

# Define the structure groups with increased sample sizes and complexity
structure_groups = {
    "Spheres": generate_data(10, complexity_factor=1.2),
    "Tori": generate_data(8, complexity_factor=1.3),
    "Projective Planes": generate_data(7, complexity_factor=1.5),
    "Cobordisms": generate_data(6, complexity_factor=1.6),
    "Higher Categories": generate_data(5, complexity_factor=1.7)
}

# Compute K-invariants for each structure and store results
k_invariants = {group: [calculate_k_invariant(**data) for data in samples]
                for group, samples in structure_groups.items()}

# Calculate summary statistics with bootstrapping for more robust estimates
summary_stats = {}
for group, values in k_invariants.items():
    mean_val = np.mean(values)
    std_dev = np.std(values, ddof=1)  # sample standard deviation
    # Bootstrapping confidence interval for mean
    boot_result = bootstrap((values,), np.mean, confidence_level=0.95, n_resamples=1000)
    conf_int = boot_result.confidence_interval
    summary_stats[group] = {
        'Mean': mean_val,
        'Std Dev': std_dev,
        'Confidence Interval': conf_int
    }
    logger.info(f'{group} - Mean: {mean_val}, Std Dev: {std_dev}, 95% CI: {conf_int}')

# Perform pairwise t-tests between each group with enhanced error handling
pairwise_t_tests = {}
for (group1, values1), (group2, values2) in itertools.combinations(k_invariants.items(), 2):
    try:
        t_stat, p_val = ttest_ind(values1, values2, equal_var=False)  # Welch's t-test
        pairwise_t_tests[(group1, group2)] = (t_stat, p_val)
        logger.info(f'T-test {group1} vs {group2}: t-statistic = {t_stat}, p-value = {p_val}')
    except Exception as e:
        logger.error(f'Error in T-test between {group1} and {group2}: {e}')
        pairwise_t_tests[(group1, group2)] = (np.nan, np.nan)

# Conduct ANOVA test across all groups
try:
    f_stat, p_val = f_oneway(*(values for values in k_invariants.values()))
    logger.info(f'ANOVA Test: F-statistic = {f_stat}, p-value = {p_val}')
except Exception as e:
    logger.error(f'Error in ANOVA test: {e}')
    f_stat, p_val = np.nan, np.nan

# Summary output
logger.info("\nSummary Statistics and Pairwise Comparisons:")
for group, stats in summary_stats.items():
    logger.info(f"{group}: {stats}")
logger.info("\nPairwise T-Test Results:")
for groups, (t_stat, p_val) in pairwise_t_tests.items():
    logger.info(f"{groups}: t-statistic = {t_stat}, p-value = {p_val}")
logger.info(f"\nANOVA Test: F-statistic = {f_stat}, p-value = {p_val}")

# Results dictionary for further use or analysis
results = {
    'summary_stats': summary_stats,
    'pairwise_t_tests': pairwise_t_tests,
    'anova_test': {'F-statistic': f_stat, 'p-value': p_val}
}
