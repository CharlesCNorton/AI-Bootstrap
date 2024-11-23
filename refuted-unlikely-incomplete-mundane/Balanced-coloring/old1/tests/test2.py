import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

# Probabilistic Analysis Setup
# Goal: Use probabilistic methods to test if the discrepancy sequence D_n can be considered bounded.
# We'll simulate discrepancies as a stochastic process and compare it to bounds from concentration inequalities.

# Parameters
num_simulations = 8000  # Number of independent simulations of the discrepancy sequence
n_steps = 300000  # Number of steps for each simulation to observe the behavior of discrepancies
initial_discrepancy = 0.0

# Arrays to hold discrepancy values over multiple simulations
discrepancy_values = []

# Simulate Discrepancy Sequence as a Random Walk (Stochastic Process)
for sim in range(num_simulations):
    discrepancy = initial_discrepancy
    discrepancies = []

    for step in range(1, n_steps + 1):
        # Random probabilistic adjustment simulating coloring decisions
        # Adjustment is chosen as ±1/step to reflect probabilistic balancing
        adjustment = random.choice([-1, 1]) * (1 / step)
        discrepancy += adjustment
        discrepancies.append(discrepancy)

    discrepancy_values.append(discrepancies)

# Convert to NumPy array for easier manipulation
discrepancy_values = np.array(discrepancy_values)

# Compute mean and standard deviation across simulations
mean_discrepancy = np.mean(discrepancy_values, axis=0)
std_discrepancy = np.std(discrepancy_values, axis=0)

# Apply Azuma-Hoeffding Bound for Concentration Analysis
# Using the normal approximation to establish the confidence interval at 95% level
# Azuma-Hoeffding is applied by assuming random variables are bounded within ±1/n
confidence_level = 0.95
z_value = norm.ppf(1 - (1 - confidence_level) / 2)  # Corresponds to 95% confidence for two-sided
azuma_bound = z_value * std_discrepancy / np.sqrt(n_steps)

# Plotting Mean Discrepancy with Azuma-Hoeffding Bound
plt.figure(figsize=(14, 8))
plt.plot(range(n_steps), mean_discrepancy, label="Mean Discrepancy", color='blue')
plt.fill_between(range(n_steps), mean_discrepancy - azuma_bound, mean_discrepancy + azuma_bound, color='lightblue', alpha=0.5, label=f"Azuma-Hoeffding Bound ({int(confidence_level * 100)}%)")
plt.xlabel('Step')
plt.ylabel('Discrepancy')
plt.title('Mean Discrepancy and Azuma-Hoeffding Bound Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Additional Analysis: Displaying Summary Statistics
print("Final Mean Discrepancy:", mean_discrepancy[-1])
print("Final Standard Deviation of Discrepancy:", std_discrepancy[-1])
print("Azuma-Hoeffding Bound at Final Step (Approximate 95% Confidence):", azuma_bound[-1])
