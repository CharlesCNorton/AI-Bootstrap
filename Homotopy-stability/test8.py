import numpy as np
import matplotlib.pyplot as plt

# Redefine the original functions for Loop Space (L_n), Product Type (P_n), and Fibration Type (F_n)
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

# Define a function to clamp values within a given range to avoid issues with invalid power operations
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Set parameters for the highly rigorous naturalistic test
np.random.seed(123)  # Seed for reproducibility
num_trials_rigorous = 2000  # Increased number of trials for each homotopy level and initial condition
n_rigorous_values = list(range(1, 31))  # Extended homotopy levels (n = 1 to 30)
a0_values_rigorous = np.random.uniform(0.1, 5.0, 10)  # Randomly sampled initial conditions for a0 from a larger domain

# Generate more types of random perturbations
epsilon_normal_rigorous = np.random.normal(0, 1.5, num_trials_rigorous)  # Normal distribution (mean=0, std=1.5)
epsilon_uniform_rigorous = np.random.uniform(-1.5, 1.5, num_trials_rigorous)  # Uniform distribution for broader random perturbations
epsilon_laplace_rigorous = np.random.laplace(0, 1.0, num_trials_rigorous)  # Laplace distribution (mean=0, scale=1.0)

# Run the rigorous naturalistic test and collect results
rigorous_results = []

for a0 in a0_values_rigorous:
    for n in n_rigorous_values:
        for epsilon in epsilon_normal_rigorous:
            # Clamp epsilon to stay within a valid range for fractional powers
            epsilon_clamped = clamp(epsilon, -0.9, 0.9)
            epsilon_scaled = epsilon_clamped / (1 + n)  # Adaptive scaling

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under clamped random epsilon (normal distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                rigorous_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                rigorous_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN for statistical reporting

        for epsilon in epsilon_uniform_rigorous:
            # Clamp epsilon to stay within valid range
            epsilon_clamped = clamp(epsilon, -0.9, 0.9)
            epsilon_scaled = epsilon_clamped / (1 + n)

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under clamped random epsilon (uniform distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                rigorous_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                rigorous_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN

        for epsilon in epsilon_laplace_rigorous:
            # Clamp epsilon to stay within valid range
            epsilon_clamped = clamp(epsilon, -0.9, 0.9)
            epsilon_scaled = epsilon_clamped / (1 + n)

            # Fibration Type F_n with cup products and adaptive epsilon for different a0 values under clamped random epsilon (Laplace distribution)
            try:
                F_n_cup_value = fibration_type_F_n_with_cup_products(a0, a0, a0, epsilon_scaled, n)
                rigorous_results.append((a0, n, epsilon_scaled, F_n_cup_value))
            except (ValueError, RuntimeWarning):
                rigorous_results.append((a0, n, epsilon_scaled, np.nan))  # Track invalid results as NaN

# Summarize results for visualization and statistical analysis
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plotting statistical analysis for highly rigorous naturalistic test
mean_values_rigorous = []
std_devs_rigorous = []
unstable_percentages = []
medians_rigorous = []
iqr_values_rigorous = []

for n in n_rigorous_values:
    f_values = [result[3] for result in rigorous_results if result[1] == n]
    valid_values = [v for v in f_values if not np.isnan(v)]

    mean_values_rigorous.append(np.mean(valid_values))
    std_devs_rigorous.append(np.std(valid_values))
    medians_rigorous.append(np.median(valid_values))
    iqr_values_rigorous.append(np.percentile(valid_values, 75) - np.percentile(valid_values, 25))

    # Calculate percentage of unstable (NaN) results
    unstable_count = np.sum(np.isnan(f_values))
    unstable_percentage = (unstable_count / len(f_values)) * 100
    unstable_percentages.append(unstable_percentage)

# Plot mean ± std deviation and instability percentage
ax.errorbar(n_rigorous_values, mean_values_rigorous, yerr=std_devs_rigorous, fmt='-o', label='Mean ± Std Dev of F_n (Rigorous)', capsize=5)
ax.plot(n_rigorous_values, unstable_percentages, 'r-', label='Percentage of Unstable Trials (NaN)', marker='x')
ax.set_xlabel("Homotopy Level n")
ax.set_ylabel("F_n Value and Instability Percentage")
ax.set_title("Highly Rigorous Naturalistic Test: Stability and Instability Analysis under Extreme Perturbations")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Print statistical summary
print("Statistical Summary of Highly Rigorous Naturalistic Test:")
print(f"{'Homotopy Level':<20}{'Mean':<10}{'Std Dev':<10}{'Median':<10}{'IQR':<10}{'% Unstable':<15}")
for i, n in enumerate(n_rigorous_values):
    print(f"{n:<20}{mean_values_rigorous[i]:<10.4f}{std_devs_rigorous[i]:<10.4f}{medians_rigorous[i]:<10.4f}{iqr_values_rigorous[i]:<10.4f}{unstable_percentages[i]:<15.2f}")
