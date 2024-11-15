import numpy as np
import scipy.stats as sp_stats  # Explicit alias to avoid conflicts
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to generate synthetic data with high complexity
def generate_data(n, complexity_factor=1.0):
    return [
        {
            "c_values": (complexity_factor * np.random.uniform(1, 5, 100)).tolist(),
            "topological_structures": (complexity_factor * np.random.exponential(5, 100)).tolist()
        }
        for _ in range(n)
    ]

# Enhanced data categories with higher sample size
data = {
    "Spheres": generate_data(1000, complexity_factor=1.2),
    "Tori": generate_data(1000, complexity_factor=2.1),
    "Projective Planes": generate_data(1000, complexity_factor=3.0),
    "Cobordisms": generate_data(1000, complexity_factor=3.5),
    "Higher Categories": generate_data(1000, complexity_factor=5.0),
}

# Calculating summary statistics
summary_stats = {}
for key, values in data.items():
    combined_values = np.array([val for sublist in values for val in sublist["c_values"]])
    mean = np.mean(combined_values)
    std_dev = np.std(combined_values, ddof=1)
    ci = sp_stats.t.interval(0.95, len(combined_values) - 1, loc=mean, scale=std_dev / np.sqrt(len(combined_values)))
    summary_stats[key] = {"Mean": mean, "Std Dev": std_dev, "Confidence Interval": ci}

# Output summary statistics
for category, stats in summary_stats.items():
    print(f"{category} - Mean: {stats['Mean']}, Std Dev: {stats['Std Dev']}, 95% CI: {stats['Confidence Interval']}")

# Pairwise T-Tests with Bonferroni correction for multiple comparisons
categories = list(data.keys())
num_comparisons = len(categories) * (len(categories) - 1) // 2
bonferroni_correction = 0.05 / num_comparisons

for i, cat1 in enumerate(categories):
    for cat2 in categories[i + 1:]:
        combined_values_1 = np.array([val for sublist in data[cat1] for val in sublist["c_values"]])
        combined_values_2 = np.array([val for sublist in data[cat2] for val in sublist["c_values"]])
        t_stat, p_value = ttest_ind(combined_values_1, combined_values_2, equal_var=False)
        corrected_p_value = min(p_value * num_comparisons, 1)  # Apply Bonferroni correction
        print(f"T-test {cat1} vs {cat2}: t-statistic = {t_stat}, corrected p-value = {corrected_p_value}")

# ANOVA Test using `sp_stats.f_oneway`
group_values = [np.array([val for sublist in data[category] for val in sublist["c_values"]]) for category in categories]
f_stat, p_val = sp_stats.f_oneway(*group_values)
print(f"ANOVA Test: F-statistic = {f_stat}, p-value = {p_val}")

# Tukey HSD Test for multiple comparisons
flattened_data = [(category, val) for category, category_values in data.items() for sublist in category_values for val in sublist["c_values"]]
categories_flat, values_flat = zip(*flattened_data)
tukey_test = pairwise_tukeyhsd(endog=np.array(values_flat), groups=np.array(categories_flat), alpha=0.05)
print(tukey_test)
