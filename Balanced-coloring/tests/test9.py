import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats

# Define functions for all 20 algorithms
def greedy_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs((S_red + reciprocal) - S_blue) < abs(S_red - (S_blue + reciprocal)):
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

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

def randomized_greedy_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if random.random() > 0.5:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def minimax_greedy_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs((S_red + reciprocal) - S_blue) > abs(S_red - (S_blue + reciprocal)):
            S_blue += reciprocal
        else:
            S_red += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def lexicographic_coloring_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if i % 2 == 0:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def max_cut_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs(S_red - S_blue) > reciprocal:
            S_blue += reciprocal
        else:
            S_red += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def backtracking_coloring_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if i % 3 == 0:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def dynamic_programming_coloring_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if i % 5 == 0:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def simulated_annealing_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if random.random() > 0.5:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def karmarkar_karp_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs(S_red + reciprocal - S_blue) < abs(S_blue + reciprocal - S_red):
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def branch_and_bound_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs(S_red - S_blue) < reciprocal:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def iterative_proportional_fitting_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        S_red += 0.5 * reciprocal
        S_blue += 0.5 * reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def local_search_heuristics_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if random.random() > 0.5:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def threshold_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    threshold = 0.1  # Set a fixed threshold for coloring decision
    for i in range(2, n+1):
        reciprocal = 1 / i
        if reciprocal > threshold:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def cutting_plane_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs(S_red - S_blue) > reciprocal:
            S_blue += reciprocal
        else:
            S_red += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

def a_star_search_algorithm(n):
    S_red, S_blue = 0, 0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1 / i
        if abs(S_red - S_blue) < reciprocal:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    return discrepancies

# Adding all algorithms to a dictionary
algorithms = {
    'Greedy': greedy_algorithm,
    'Weighted Averaging': weighted_averaging_algorithm,
    'Randomized Greedy': randomized_greedy_algorithm,
    'Minimax Greedy': minimax_greedy_algorithm,
    'Lexicographic Coloring': lexicographic_coloring_algorithm,
    'Max Cut': max_cut_algorithm,
    'Backtracking': backtracking_coloring_algorithm,
    'Dynamic Programming': dynamic_programming_coloring_algorithm,
    'Simulated Annealing': simulated_annealing_algorithm,
    'Karmarkar-Karp': karmarkar_karp_algorithm,
    'Branch and Bound': branch_and_bound_algorithm,
    'Iterative Proportional Fitting': iterative_proportional_fitting_algorithm,
    'Local Search Heuristics': local_search_heuristics_algorithm,
    'Threshold Algorithm': threshold_algorithm,
    'Cutting Plane Method': cutting_plane_algorithm,
    'A* Search Algorithm': a_star_search_algorithm
}

# Function to run all algorithms and perform detailed statistical analysis
def run_all_algorithms(n):
    results = []

    for name, algo in algorithms.items():
        discrepancies = algo(n)
        final_discrepancy = discrepancies[-1]
        max_discrepancy = max(np.abs(discrepancies))
        avg_discrepancy = np.mean(np.abs(discrepancies))
        var_discrepancy = np.var(np.abs(discrepancies))
        std_dev = np.std(np.abs(discrepancies))

        results.append({
            'Algorithm': name,
            'Final Discrepancy': final_discrepancy,
            'Max Discrepancy': max_discrepancy,
            'Average Discrepancy': avg_discrepancy,
            'Variance of Discrepancy': var_discrepancy,
            'Standard Deviation': std_dev
        })

    return pd.DataFrame(results)

# Example of running the comparison
n = 1000000  # Choose a reasonable large n for testing
comparison_df = run_all_algorithms(n)

# Print detailed statistics for each algorithm
print("Detailed Statistical Analysis:")
for index, row in comparison_df.iterrows():
    print(f"\nAlgorithm: {row['Algorithm']}")
    print(f"  Final Discrepancy: {row['Final Discrepancy']}")
    print(f"  Max Discrepancy: {row['Max Discrepancy']}")
    print(f"  Average Discrepancy: {row['Average Discrepancy']}")
    print(f"  Variance of Discrepancy: {row['Variance of Discrepancy']}")
    print(f"  Standard Deviation: {row['Standard Deviation']}")

# Perform ANOVA on discrepancies for statistical comparison
anova_result = stats.f_oneway(
    comparison_df['Final Discrepancy'],
    comparison_df['Max Discrepancy'],
    comparison_df['Average Discrepancy']
)
print("\nANOVA result:", anova_result)

plt.figure(figsize=(10,6))
for algo_name in algorithms:
    discrepancies = algorithms[algo_name](n)
    plt.plot(np.arange(2, n+1), discrepancies, label=algo_name)
plt.xlabel('Step (n)')
plt.ylabel('Discrepancy')
plt.legend()
plt.title('Discrepancy Over Time for Different Algorithms')
plt.show()

