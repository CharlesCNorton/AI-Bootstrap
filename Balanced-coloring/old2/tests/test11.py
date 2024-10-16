import numpy as np
import time
import matplotlib.pyplot as plt

def greedy_algorithm(n):
    S_red = 0.0
    S_blue = 0.0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1.0 / i
        if S_red < S_blue:
            S_red += reciprocal
        else:
            S_blue += reciprocal
        discrepancies.append(S_red - S_blue)
    final_discrepancy = S_red - S_blue
    average_discrepancy = np.mean(np.abs(discrepancies))
    max_discrepancy = np.max(np.abs(discrepancies))
    return final_discrepancy, average_discrepancy, max_discrepancy, discrepancies

def minimax_greedy_algorithm(n):
    # Placeholder for Minimax Greedy Algorithm
    # Assuming it assigns each reciprocal to minimize the current maximum discrepancy
    S_red = 0.0
    S_blue = 0.0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1.0 / i
        # Compute potential discrepancies
        D_red = S_red + reciprocal - S_blue
        D_blue = S_red - (S_blue + reciprocal)
        # Choose the assignment that minimizes the maximum discrepancy
        if abs(D_red) < abs(D_blue):
            S_red += reciprocal
            discrepancies.append(D_red)
        else:
            S_blue += reciprocal
            discrepancies.append(D_blue)
    final_discrepancy = S_red - S_blue
    average_discrepancy = np.mean(np.abs(discrepancies))
    max_discrepancy = np.max(np.abs(discrepancies))
    return final_discrepancy, average_discrepancy, max_discrepancy, discrepancies

def dynamic_partitioning_algorithm(n):
    # Placeholder for Dynamic Partitioning Algorithm
    # For illustration, let's implement a simple dynamic approach where weights are adjusted
    S_red = 0.0
    S_blue = 0.0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1.0 / i
        # Calculate the proportion to assign based on current discrepancy
        if S_red < S_blue:
            weight_red = 0.7
            weight_blue = 0.3
        else:
            weight_red = 0.3
            weight_blue = 0.7
        S_red += weight_red * reciprocal
        S_blue += weight_blue * reciprocal
        discrepancies.append(S_red - S_blue)
    final_discrepancy = S_red - S_blue
    average_discrepancy = np.mean(np.abs(discrepancies))
    max_discrepancy = np.max(np.abs(discrepancies))
    return final_discrepancy, average_discrepancy, max_discrepancy, discrepancies

def weighted_averaging_algorithm(n):
    S_red = 0.0
    S_blue = 0.0
    discrepancies = []
    for i in range(2, n+1):
        reciprocal = 1.0 / i
        if S_red < S_blue:
            S_red += 0.6 * reciprocal
            S_blue += 0.4 * reciprocal
        else:
            S_red += 0.4 * reciprocal
            S_blue += 0.6 * reciprocal
        discrepancies.append(S_red - S_blue)
    final_discrepancy = S_red - S_blue
    average_discrepancy = np.mean(np.abs(discrepancies))
    max_discrepancy = np.max(np.abs(discrepancies))
    return final_discrepancy, average_discrepancy, max_discrepancy, discrepancies

def run_comparison(n):
    results = {}
    algorithms = {
        'Greedy': greedy_algorithm,
        'Minimax Greedy': minimax_greedy_algorithm,
        'Dynamic Partitioning': dynamic_partitioning_algorithm,
        'Weighted Averaging': weighted_averaging_algorithm
    }

    for name, func in algorithms.items():
        print(f"Running {name} Algorithm...")
        start_time = time.time()
        final_disc, avg_disc, max_disc, discrepancies = func(n)
        end_time = time.time()
        exec_time = end_time - start_time
        results[name] = {
            'Final Discrepancy': final_disc,
            'Average Discrepancy': avg_disc,
            'Maximum Discrepancy': max_disc,
            'Execution Time (s)': exec_time,
            'Discrepancies': discrepancies  # For potential further analysis
        }
        print(f"{name} completed in {exec_time:.2f} seconds.")
        print(f"Final Discrepancy: {final_disc}")
        print(f"Average Discrepancy: {avg_disc}")
        print(f"Maximum Discrepancy: {max_disc}\n")

    return results

def statistical_analysis(results):
    print("Statistical Analysis of Algorithms:\n")
    for name, metrics in results.items():
        print(f"Algorithm: {name}")
        print(f"  Final Discrepancy: {metrics['Final Discrepancy']}")
        print(f"  Average Discrepancy: {metrics['Average Discrepancy']}")
        print(f"  Maximum Discrepancy: {metrics['Maximum Discrepancy']}")
        print(f"  Execution Time (s): {metrics['Execution Time (s)']}\n")

    # Optional: Plotting Maximum Discrepancy
    plt.figure(figsize=(10, 6))
    for name, metrics in results.items():
        plt.plot(metrics['Discrepancies'], label=name)
    plt.xlabel('Integer i')
    plt.ylabel('Discrepancy D_n')
    plt.title('Discrepancy Over Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n = 1_000_000_000
    results = run_comparison(n)
    statistical_analysis(results)
