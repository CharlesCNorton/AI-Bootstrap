import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import multiprocessing
import random

def greedy_algorithm(n):
    red_sum = 0.0
    blue_sum = 0.0
    discrepancies = []
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if red_sum < blue_sum:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
        discrepancy = abs(red_sum - blue_sum)
        discrepancies.append(discrepancy)
    return red_sum, blue_sum, discrepancies

def weighted_averaging_discrete(n, red_weight=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    red_sum = 0.0
    blue_sum = 0.0
    discrepancies = []
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if random.random() < red_weight:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
        discrepancy = abs(red_sum - blue_sum)
        discrepancies.append(discrepancy)
    return red_sum, blue_sum, discrepancies

def run_experiments():
    n_values = [10**i for i in range(3, 8)]  # n = 1e3 to 1e7
    greedy_discrepancies = []
    weight_avg_discrepancies = []
    weight_avg_6040_discrepancies = []

    greedy_times = []
    weight_avg_times = []
    weight_avg_6040_times = []

    print("Running experiments for various n values...\n")
    for n in n_values:
        print(f"n = {n}")
        # Greedy Algorithm
        start_time = time.time()
        _, _, discrepancies = greedy_algorithm(n)
        end_time = time.time()
        greedy_time = end_time - start_time
        greedy_times.append(greedy_time)
        greedy_discrepancies.append(discrepancies[-1])
        print(f"Greedy Algorithm:\nDiscrepancy = {discrepancies[-1]:.8e}, Time = {greedy_time:.4f} s")

        # Weighted Averaging Algorithm (50/50 Split)
        start_time = time.time()
        _, _, discrepancies = weighted_averaging_discrete(n, red_weight=0.5, seed=42)
        end_time = time.time()
        weight_avg_time = end_time - start_time
        weight_avg_times.append(weight_avg_time)
        weight_avg_discrepancies.append(discrepancies[-1])
        print(f"Weighted Averaging Algorithm (50/50):\nDiscrepancy = {discrepancies[-1]:.8e}, Time = {weight_avg_time:.4f} s")

        # Weighted Averaging Algorithm (60/40 Split)
        start_time = time.time()
        _, _, discrepancies = weighted_averaging_discrete(n, red_weight=0.6, seed=42)
        end_time = time.time()
        weight_avg_6040_time = end_time - start_time
        weight_avg_6040_times.append(weight_avg_6040_time)
        weight_avg_6040_discrepancies.append(discrepancies[-1])
        print(f"Weighted Averaging Algorithm (60/40):\nDiscrepancy = {discrepancies[-1]:.8e}, Time = {weight_avg_6040_time:.4f} s")

        print("-" * 60)

    # Analyze and plot results
    analyze_results(n_values, greedy_discrepancies, weight_avg_discrepancies, weight_avg_6040_discrepancies,
                    greedy_times, weight_avg_times, weight_avg_6040_times)

def analyze_results(n_values, greedy_discrepancies, weight_avg_discrepancies, weight_avg_6040_discrepancies,
                    greedy_times, weight_avg_times, weight_avg_6040_times):
    # Convert lists to numpy arrays for convenience
    n_values = np.array(n_values, dtype=np.float64)
    greedy_discrepancies = np.array(greedy_discrepancies)
    weight_avg_discrepancies = np.array(weight_avg_discrepancies)
    weight_avg_6040_discrepancies = np.array(weight_avg_6040_discrepancies)

    # Log-log plots to determine the relationship between D(n) and n
    plt.figure(figsize=(12, 8))
    plt.loglog(n_values, greedy_discrepancies, 'o-', label='Greedy Algorithm')
    plt.loglog(n_values, weight_avg_discrepancies, 's-', label='Weighted Averaging (50/50)')
    plt.loglog(n_values, weight_avg_6040_discrepancies, 'd-', label='Weighted Averaging (60/40)')
    plt.xlabel('n')
    plt.ylabel('Discrepancy D(n)')
    plt.title('Discrepancy vs. n on Log-Log Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig('discrepancy_vs_n_loglog.png')
    plt.show()

    # Linear regression on log-log data to estimate the exponent alpha in D(n) ~ n^(-alpha)
    def power_law(n, a, alpha):
        return a * n ** (-alpha)

    print("\nPerforming regression analysis on the discrepancies...\n")

    # Greedy Algorithm regression
    popt_greedy, pcov_greedy = curve_fit(power_law, n_values, greedy_discrepancies)
    a_greedy, alpha_greedy = popt_greedy
    print(f"Greedy Algorithm Discrepancy ~ {a_greedy:.4e} * n^(-{alpha_greedy:.4f})")

    # Weighted Averaging Algorithm (50/50) regression
    popt_weight_avg, pcov_weight_avg = curve_fit(power_law, n_values, weight_avg_discrepancies)
    a_weight_avg, alpha_weight_avg = popt_weight_avg
    print(f"Weighted Averaging (50/50) Discrepancy ~ {a_weight_avg:.4e} * n^(-{alpha_weight_avg:.4f})")

    # Weighted Averaging Algorithm (60/40) regression
    popt_weight_avg_6040, pcov_weight_avg_6040 = curve_fit(power_law, n_values, weight_avg_6040_discrepancies)
    a_weight_avg_6040, alpha_weight_avg_6040 = popt_weight_avg_6040
    print(f"Weighted Averaging (60/40) Discrepancy ~ {a_weight_avg_6040:.4e} * n^(-{alpha_weight_avg_6040:.4f})")

    # Execution Times Plot
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, greedy_times, 'o-', label='Greedy Algorithm')
    plt.plot(n_values, weight_avg_times, 's-', label='Weighted Averaging (50/50)')
    plt.plot(n_values, weight_avg_6040_times, 'd-', label='Weighted Averaging (60/40)')
    plt.xlabel('n')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs. n')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time_vs_n.png')
    plt.show()

    # Interpretation of results
    print("\nInterpretation of Results:")
    print("-" * 50)
    print("Greedy Algorithm:")
    print(f"Estimated exponent alpha = {alpha_greedy:.4f}")
    if alpha_greedy > 0:
        print("Discrepancy decreases with increasing n.")
    else:
        print("Discrepancy does not decrease with increasing n.")
    print("\nWeighted Averaging Algorithm (50/50):")
    print(f"Estimated exponent alpha = {alpha_weight_avg:.4f}")
    if alpha_weight_avg > 0:
        print("Discrepancy decreases with increasing n.")
    else:
        print("Discrepancy does not decrease with increasing n.")
    print("\nWeighted Averaging Algorithm (60/40):")
    print(f"Estimated exponent alpha = {alpha_weight_avg_6040:.4f}")
    if alpha_weight_avg_6040 > 0:
        print("Discrepancy decreases with increasing n.")
    else:
        print("Discrepancy does not decrease with increasing n.")

    # Display discrepancies and times in a table format
    print("\nSummary of Results:")
    print("-" * 50)
    print(f"{'n':>10} | {'Greedy D(n)':>15} | {'WeightAvg D(n)':>15} | {'WeightAvg60/40 D(n)':>20} | {'Greedy Time (s)':>15}")
    print("-" * 100)
    for i in range(len(n_values)):
        print(f"{int(n_values[i]):>10} | {greedy_discrepancies[i]:>15.8e} | {weight_avg_discrepancies[i]:>15.8e} | {weight_avg_6040_discrepancies[i]:>20.8e} | {greedy_times[i]:>15.4f}")
    print("-" * 100)

if __name__ == "__main__":
    run_experiments()
