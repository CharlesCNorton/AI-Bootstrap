import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Actual data from results for each attractor
z_values = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
attractor_alpha_counts = np.array([4592, 9543, 14632, 19817, 25057, 30344, 35691, 41054, 46465, 51900])
attractor_beta_counts = np.array([4159, 8514, 12941, 17425, 21936, 26491, 31062, 35647, 40262, 44878])
attractor_gamma_counts = np.array([2752, 5978, 9361, 12880, 16500, 20137, 23878, 27652, 31485, 35343])

# Function to model power-law growth
def power_law(z, a, exponent):
    return a * z ** exponent

# Fit empirical data with both original and new theoretical exponents
def fit_and_compare(z_values, counts, original_exponent, new_exponent, attractor_name):
    # Fit empirical data to both exponents
    popt_orig, _ = curve_fit(lambda z, a: power_law(z, a, original_exponent), z_values, counts)
    popt_new, _ = curve_fit(lambda z, a: power_law(z, a, new_exponent), z_values, counts)

    # Generate predictions based on both fits
    fitted_original = power_law(z_values, popt_orig[0], original_exponent)
    fitted_new = power_law(z_values, popt_new[0], new_exponent)

    # Plot empirical data vs. fits
    plt.figure(figsize=(8, 6))
    plt.scatter(z_values, counts, label='Empirical Data', color='blue')
    plt.plot(z_values, fitted_original, label=f'Original Exponent ({original_exponent})', linestyle='--', color='green')
    plt.plot(z_values, fitted_new, label=f'New Exponent ({new_exponent})', linestyle='-', color='red')
    plt.title(f'Attractor {attractor_name} Growth Comparison')
    plt.xlabel('z value')
    plt.ylabel('Number of Solutions')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Calculate residuals (how well each model fits the data)
    residuals_original = np.sum((counts - fitted_original) ** 2)
    residuals_new = np.sum((counts - fitted_new) ** 2)

    print(f"Residuals for attractor {attractor_name}:")
    print(f"Original exponent ({original_exponent}): {residuals_original}")
    print(f"New exponent ({new_exponent}): {residuals_new}")
    print("\n")

# Compare the fits for each attractor
fit_and_compare(z_values, attractor_alpha_counts, 0.5, 1.05, 'α')
fit_and_compare(z_values, attractor_beta_counts, 0.5, 1.03, 'β')
fit_and_compare(z_values, attractor_gamma_counts, 0.45, 1.11, 'γ')
