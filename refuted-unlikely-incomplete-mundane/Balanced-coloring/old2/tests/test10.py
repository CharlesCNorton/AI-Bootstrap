import numpy as np
import matplotlib.pyplot as plt

# Define the weighted averaging algorithm
def weighted_averaging_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if S_red < S_blue:
            S_red += 0.6 * reciprocal
            S_blue += 0.4 * reciprocal
        else:
            S_red += 0.4 * reciprocal
            S_blue += 0.6 * reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

# Set a smaller value for n
n = 10**9

# Run the weighted averaging algorithm
discrepancies = weighted_averaging_algorithm(n)

# Calculate detailed statistics
final_discrepancy = discrepancies[-1]
max_discrepancy = max(np.abs(discrepancies))
avg_discrepancy = np.mean(np.abs(discrepancies))
var_discrepancy = np.var(np.abs(discrepancies))
std_dev = np.std(np.abs(discrepancies))

# Print detailed statistics
print("Detailed Statistical Analysis:")
print(f"  Final Discrepancy: {final_discrepancy}")
print(f"  Max Discrepancy: {max_discrepancy}")
print(f"  Average Discrepancy: {avg_discrepancy}")
print(f"  Variance of Discrepancy: {var_discrepancy}")
print(f"  Standard Deviation: {std_dev}")

# Plotting the discrepancy over time
plt.figure(figsize=(10, 6))
plt.plot(np.arange(2, n+1), discrepancies, label='Weighted Averaging Algorithm')
plt.xlabel('Step (n)')
plt.ylabel('Discrepancy')
plt.legend()
plt.title('Discrepancy Over Time for Weighted Averaging Algorithm')
plt.show()
