import numpy as np
from scipy.stats import ttest_1samp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Constants
NUM_SAMPLES = 1000
DIMENSIONS = range(1, 31)  # d = 1 to 30
NUM_BOOTSTRAPS = 10000
ALPHA = 0.05  # Significance level for confidence intervals

# Seed for reproducibility
np.random.seed(42)

def path_space(x, y, d):
    """
    Constructs the path space matrix P(d)(x, y).
    """
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * d)
    M = np.random.uniform(-1, 1, (d, d))
    return np.eye(d) + epsilon * np.exp(-0.3 * dist) * M

def measure_reflexivity(d):
    """
    Measures reflexivity R(d) by averaging the normalized similarity of (P(d)(x, x) - I).
    """
    similarities = []
    for _ in range(NUM_SAMPLES):
        x = np.random.uniform(-1, 1, d)
        P = path_space(x, x, d)
        norm_diff = np.linalg.norm(P - np.eye(d), 'fro')
        norm_I = np.linalg.norm(np.eye(d), 'fro')
        R = 1 - (norm_diff / norm_I)
        similarities.append(R)
    return similarities

def measure_symmetry(d):
    """
    Measures symmetry S(d) by averaging the normalized similarity between (P(d)(x, y) and P(d)(y, x)).
    """
    similarities = []
    for _ in range(NUM_SAMPLES):
        x = np.random.uniform(-1, 1, d)
        y = np.random.uniform(-1, 1, d)
        P_xy = path_space(x, y, d)
        P_yx = path_space(y, x, d)
        norm_diff = np.linalg.norm(P_xy - P_yx, 'fro')
        norm_sum = np.linalg.norm(P_xy, 'fro') + np.linalg.norm(P_yx, 'fro')
        if norm_sum == 0:
            S = 1  # Both matrices are zero; define as perfectly symmetric
        else:
            S = 1 - (norm_diff / norm_sum)
        similarities.append(S)
    return similarities

def measure_transitivity(d):
    """
    Measures transitivity T(d) by averaging the normalized similarity of (P(d)(x, y) * P(d)(y, z) and P(d)(x, z)).
    """
    similarities = []
    for _ in range(NUM_SAMPLES):
        x = np.random.uniform(-1, 1, d)
        y = np.random.uniform(-1, 1, d)
        z = np.random.uniform(-1, 1, d)
        P_xy = path_space(x, y, d)
        P_yz = path_space(y, z, d)
        P_xz = path_space(x, z, d)
        composition = np.dot(P_xy, P_yz)
        norm_diff = np.linalg.norm(composition - P_xz, 'fro')
        norm_sum = np.linalg.norm(composition, 'fro') + np.linalg.norm(P_xz, 'fro')
        if norm_sum == 0:
            T = 1  # Both matrices are zero; define as perfectly transitive
        else:
            T = 1 - (norm_diff / norm_sum)
        similarities.append(T)
    return similarities

def bootstrap_confidence_interval(data, num_bootstrap=NUM_BOOTSTRAPS, alpha=ALPHA):
    """
    Computes the bootstrap confidence interval for the mean of the data.
    """
    boot_means = np.empty(num_bootstrap)
    n = len(data)
    for i in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

# Storage for results
results = {
    'dimension': [],
    'R_mean': [],
    'R_ci_lower': [],
    'R_ci_upper': [],
    'S_mean': [],
    'S_ci_lower': [],
    'S_ci_upper': [],
    'T_mean': [],
    'T_ci_lower': [],
    'T_ci_upper': [],
    'S/R_mean': [],
    'S/R_ci_lower': [],
    'S/R_ci_upper': []
}

# Measurement loop
for d in tqdm(DIMENSIONS, desc="Measuring properties"):
    R_values = measure_reflexivity(d)
    S_values = measure_symmetry(d)
    T_values = measure_transitivity(d)

    # Compute means
    R_mean = np.mean(R_values)
    S_mean = np.mean(S_values)
    T_mean = np.mean(T_values)

    # Compute bootstrap confidence intervals
    R_ci = bootstrap_confidence_interval(R_values)
    S_ci = bootstrap_confidence_interval(S_values)
    T_ci = bootstrap_confidence_interval(T_values)

    # Compute S/R ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        S_R_ratios = np.array(S_values) / np.array(R_values)
        # Handle cases where R_values are zero to avoid division by zero
        S_R_ratios = np.where(np.isfinite(S_R_ratios), S_R_ratios, 0)

    S_R_mean = np.mean(S_R_ratios)
    S_R_ci = bootstrap_confidence_interval(S_R_ratios)

    # Store results
    results['dimension'].append(d)
    results['R_mean'].append(R_mean)
    results['R_ci_lower'].append(R_ci[0])
    results['R_ci_upper'].append(R_ci[1])
    results['S_mean'].append(S_mean)
    results['S_ci_lower'].append(S_ci[0])
    results['S_ci_upper'].append(S_ci[1])
    results['T_mean'].append(T_mean)
    results['T_ci_lower'].append(T_ci[0])
    results['T_ci_upper'].append(T_ci[1])
    results['S/R_mean'].append(S_R_mean)
    results['S/R_ci_lower'].append(S_R_ci[0])
    results['S/R_ci_upper'].append(S_R_ci[1])

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Define theoretical asymptotic values
R_inf_theoretical = 0.954
S_inf_theoretical = 0.983
T_inf_theoretical = 0.979
gamma_d_theoretical = 0.000640  # As per THEOREM 2
ratio_base_theoretical = 1.001570

# Fit exponential decay models without initial guesses
def exponential_decay(d, P_inf, alpha, beta):
    return P_inf + alpha * np.exp(-beta * d)

# Fit for R(d)
popt_R, pcov_R = curve_fit(exponential_decay, df_results['dimension'], df_results['R_mean'], maxfev=10000)
# Fit for S(d)
popt_S, pcov_S = curve_fit(exponential_decay, df_results['dimension'], df_results['S_mean'], maxfev=10000)
# Fit for T(d)
popt_T, pcov_T = curve_fit(exponential_decay, df_results['dimension'], df_results['T_mean'], maxfev=10000)

# Define linear model for S/R ratio
def linear_model(d, a, b):
    return a + b * d

# Fit linear model for S/R ratio
popt_ratio, pcov_ratio = curve_fit(linear_model, df_results['dimension'], df_results['S/R_mean'], maxfev=10000)

# Plotting
plt.figure(figsize=(14, 10))

# Plot R with fitted curve and confidence interval
plt.subplot(3, 1, 1)
plt.plot(df_results['dimension'], df_results['R_mean'], 'o', label='Reflexivity (R) Data')
plt.plot(df_results['dimension'], exponential_decay(df_results['dimension'], *popt_R), '-', label='R(d) Fit')
plt.fill_between(df_results['dimension'], df_results['R_ci_lower'], df_results['R_ci_upper'], alpha=0.2, label='R(d) 95% CI')
plt.xlabel('Dimension (d)')
plt.ylabel('Reflexivity (R)')
plt.title('Reflexivity vs Dimension')
plt.legend()
plt.grid(True)

# Plot S with fitted curve and confidence interval
plt.subplot(3, 1, 2)
plt.plot(df_results['dimension'], df_results['S_mean'], 'o', label='Symmetry (S) Data')
plt.plot(df_results['dimension'], exponential_decay(df_results['dimension'], *popt_S), '-', label='S(d) Fit')
plt.fill_between(df_results['dimension'], df_results['S_ci_lower'], df_results['S_ci_upper'], alpha=0.2, label='S(d) 95% CI')
plt.xlabel('Dimension (d)')
plt.ylabel('Symmetry (S)')
plt.title('Symmetry vs Dimension')
plt.legend()
plt.grid(True)

# Plot T with fitted curve and confidence interval
plt.subplot(3, 1, 3)
plt.plot(df_results['dimension'], df_results['T_mean'], 'o', label='Transitivity (T) Data')
plt.plot(df_results['dimension'], exponential_decay(df_results['dimension'], *popt_T), '-', label='T(d) Fit')
plt.fill_between(df_results['dimension'], df_results['T_ci_lower'], df_results['T_ci_upper'], alpha=0.2, label='T(d) 95% CI')
plt.xlabel('Dimension (d)')
plt.ylabel('Transitivity (T)')
plt.title('Transitivity vs Dimension')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot S/R ratio with fitted linear model
plt.figure(figsize=(10, 6))
plt.plot(df_results['dimension'], df_results['S/R_mean'], 'o', label='S/R Ratio Data')
plt.plot(df_results['dimension'], linear_model(df_results['dimension'], *popt_ratio), '-', label='S/R Ratio Fit')
plt.fill_between(df_results['dimension'], df_results['S/R_ci_lower'], df_results['S/R_ci_upper'], alpha=0.2, label='S/R 95% CI')
plt.xlabel('Dimension (d)')
plt.ylabel('S/R Ratio')
plt.title('Symmetry to Reflexivity Ratio vs Dimension')
plt.legend()
plt.grid(True)
plt.show()

# Output Fitted Parameters
print("Fitted Parameters:")
print(f"R(d) = {popt_R[0]:.6f} + {popt_R[1]:.6f} * exp(-{popt_R[2]:.6f} * d)")
print(f"S(d) = {popt_S[0]:.6f} + {popt_S[1]:.6f} * exp(-{popt_S[2]:.6f} * d)")
print(f"T(d) = {popt_T[0]:.6f} + {popt_T[1]:.6f} * exp(-{popt_T[2]:.6f} * d)")
print(f"S(d)/R(d) = {popt_ratio[0]:.6f} + {popt_ratio[1]:.6f} * d")

# Hierarchical Stability Check
# We will check if S_mean > T_mean > R_mean with confidence intervals
hierarchy_stability = True
for index, row in df_results.iterrows():
    S = row['S_mean']
    T = row['T_mean']
    R = row['R_mean']
    if not (S > T > R):
        hierarchy_stability = False
        print(f"Hierarchy failed at dimension {row['dimension']}: S={S}, T={T}, R={R}")
        break

if hierarchy_stability:
    print("Hierarchy S > T > R holds across all dimensions with measured means.")
else:
    print("Hierarchy S > T > R does NOT hold across all dimensions.")

# Ratio Evolution Fit
print("\nFitted S/R Ratio Parameters:")
print(f"S(d)/R(d) = {popt_ratio[0]:.6f} + {popt_ratio[1]:.6f} * d")

# Statistical Analysis
# Compare R_mean with R_inf_theoretical using one-sample t-test
print("\nStatistical Analysis:")
R_values_all = df_results['R_mean'].values
t_stat_R, p_value_R = ttest_1samp(R_values_all, R_inf_theoretical)
print(f"One-sample t-test comparing R(d) to R_inf={R_inf_theoretical}:")
print(f"T-statistic: {t_stat_R:.4f}, P-value: {p_value_R:.4f}")
if p_value_R < ALPHA:
    print("Reject the null hypothesis: R(d) is significantly different from R_inf.")
else:
    print("Fail to reject the null hypothesis: No significant difference between R(d) and R_inf.")

# Similarly, perform t-tests for S(d) and T(d)
S_values_all = df_results['S_mean'].values
T_values_all = df_results['T_mean'].values

t_stat_S, p_value_S = ttest_1samp(S_values_all, S_inf_theoretical)
print(f"\nOne-sample t-test comparing S(d) to S_inf={S_inf_theoretical}:")
print(f"T-statistic: {t_stat_S:.4f}, P-value: {p_value_S:.4f}")
if p_value_S < ALPHA:
    print("Reject the null hypothesis: S(d) is significantly different from S_inf.")
else:
    print("Fail to reject the null hypothesis: No significant difference between S(d) and S_inf.")

t_stat_T, p_value_T = ttest_1samp(T_values_all, T_inf_theoretical)
print(f"\nOne-sample t-test comparing T(d) to T_inf={T_inf_theoretical}:")
print(f"T-statistic: {t_stat_T:.4f}, P-value: {p_value_T:.4f}")
if p_value_T < ALPHA:
    print("Reject the null hypothesis: T(d) is significantly different from T_inf.")
else:
    print("Fail to reject the null hypothesis: No significant difference between T(d) and T_inf.")
