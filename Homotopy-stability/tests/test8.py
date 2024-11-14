import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Redefine the original functions for Loop Space (L_n), Product Type (P_n), and F_n
def loop_space_L_n(a0, epsilon, n):
    """Loop Space L_n function definition."""
    return ((a0 + epsilon) / 2)**(1/n) + np.cos(n * (a0 + epsilon))

def product_type_P_n(a0_1, a0_2, epsilon, n):
    """Product Type P_n function definition."""
    return ((a0_1 + epsilon)**(1/n) + np.cos(n * (a0_1 + epsilon)) +
            (a0_2 - epsilon)**(1/n) + np.sin(n * (a0_2 - epsilon))) / 2

def fibration_type_F_n_with_cup_products(a0_base, a0_fiber1, a0_fiber2, epsilon, n):
    """Fibration Type F_n function with explicit higher-order cup product interactions."""
    cup_product_1 = np.cos(n * a0_fiber1 + epsilon / 4)  # Example higher-order interaction term
    cup_product_2 = np.sin(n * a0_fiber2 + epsilon / 6)  # Example higher-order interaction term

    return ((a0_base + epsilon)**(1/n) + np.cos(n * a0_base) +
            ((a0_fiber1 + 0.5 * epsilon)**(1/(n+1)) + np.sin(n * a0_fiber1) + cup_product_1) / 2 +
            ((a0_fiber2 + 0.25 * epsilon)**(1/(n+2)) + np.sin(n * a0_fiber2) + cup_product_2) / 2) / 2

# Set parameters for the new test without clamping
np.random.seed(789)  # Seed for reproducibility
num_trials_no_clamp = 3000  # Number of trials for each homotopy level and initial condition
n_values_no_clamp = list(range(1, 51))  # Homotopy levels (n = 1 to 50)
a0_values_no_clamp = np.random.uniform(0.1, 5.0, 15)  # Randomly sampled initial conditions for a0

# Generate random perturbations without clamping
epsilon_normal_no_clamp = np.random.normal(0, 2.0, num_trials_no_clamp)  # Normal distribution (mean=0, std=2.0)
epsilon_uniform_no_clamp = np.random.uniform(-2.0, 2.0, num_trials_no_clamp)  # Uniform distribution for random perturbations
epsilon_exponential_no_clamp = np.random.exponential(1.0, num_trials_no_clamp) - 1.0  # Shifted exponential distribution

# Run the test without clamping and collect results
no_clamp_results = []

for a0 in a0_values_no_clamp:
    for n in n_values_no_clamp:
        for epsilon in epsilon_normal_no_clamp:
            # Apply an alternative scaling method (inverse-square scaling)
            epsilon_scaled = epsilon / (1 + n**2)  # New adaptive scaling strategy without clamping

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under extreme random epsilon (normal distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                no_clamp_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                no_clamp_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN for statistical reporting

        for epsilon in epsilon_uniform_no_clamp:
            # Apply an alternative scaling method (inverse-square scaling)
            epsilon_scaled = epsilon / (1 + n**2)

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under extreme random epsilon (uniform distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                no_clamp_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                no_clamp_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN

        for epsilon in epsilon_exponential_no_clamp:
            # Apply an alternative scaling method (inverse-square scaling)
            epsilon_scaled = epsilon / (1 + n**2)

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under extreme random epsilon (exponential distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                no_clamp_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                no_clamp_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN

# Summarize results for visualization and statistical analysis
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plotting statistical analysis for the no-clamp naturalistic test
mean_values_no_clamp = []
std_devs_no_clamp = []
unstable_percentages_no_clamp = []
medians_no_clamp = []
iqr_values_no_clamp = []
skewness_values_no_clamp = []
kurtosis_values_no_clamp = []

for n in n_values_no_clamp:
    f_values = [result[3] for result in no_clamp_results if result[1] == n]
    valid_values = [v for v in f_values if not np.isnan(v)]

    mean_values_no_clamp.append(np.mean(valid_values))
    std_devs_no_clamp.append(np.std(valid_values))
    medians_no_clamp.append(np.median(valid_values))
    iqr_values_no_clamp.append(np.percentile(valid_values, 75) - np.percentile(valid_values, 25))
    skewness_values_no_clamp.append(skew(valid_values))
    kurtosis_values_no_clamp.append(kurtosis(valid_values))

    # Calculate percentage of unstable (NaN) results
    unstable_count = np.sum(np.isnan(f_values))
    unstable_percentage = (unstable_count / len(f_values)) * 100
    unstable_percentages_no_clamp.append(unstable_percentage)

# Plot mean ± std deviation and instability percentage
ax.errorbar(n_values_no_clamp, mean_values_no_clamp, yerr=std_devs_no_clamp, fmt='-o', label='Mean ± Std Dev of F_n (No Clamp)', capsize=5)
ax.plot(n_values_no_clamp, unstable_percentages_no_clamp, 'r-', label='Percentage of Unstable Trials (NaN)', marker='x')
ax.set_xlabel("Homotopy Level n")
ax.set_ylabel("F_n Value and Instability Percentage")
ax.set_title("Naturalistic Test without Clamping: Stability and Instability Analysis under Extreme Perturbations")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Print statistical summary for no-clamp analysis
print("Statistical Summary of No-Clamp Naturalistic Test:")
print(f"{'Homotopy Level':<20}{'Mean':<10}{'Std Dev':<10}{'Median':<10}{'IQR':<10}{'Skewness':<10}{'Kurtosis':<10}{'% Unstable':<15}")
for i, n in enumerate(n_values_no_clamp):
    print(f"{n:<20}{mean_values_no_clamp[i]:<10.4f}{std_devs_no_clamp[i]:<10.4f}{medians_no_clamp[i]:<10.4f}{iqr_values_no_clamp[i]:<10.4f}{skewness_values_no_clamp[i]:<10.4f}{kurtosis_values_no_clamp[i]:<10.4f}{unstable_percentages_no_clamp[i]:<15.2f}")
