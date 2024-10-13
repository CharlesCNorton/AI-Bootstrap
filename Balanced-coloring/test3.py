import numpy as np
import matplotlib.pyplot as plt
import time

# Deterministic Greedy Coloring for Balancing Reciprocals
# This implementation focuses on empirical validation of the deterministic strategy.
# It tracks the discrepancy, the potential function, and the energy function to ensure boundedness.

# Greedy Coloring Strategy
def greedy_coloring(n):
    red_sum, blue_sum = 0.0, 0.0
    red_set, blue_set = [], []

    discrepancy_history = []
    energy_history = []
    potential_history = []

    for i in range(2, n + 1):
        # Determine which choice minimizes the discrepancy
        if abs((red_sum + 1 / i) - blue_sum) < abs((blue_sum + 1 / i) - red_sum):
            red_set.append(i)
            red_sum += 1 / i
        else:
            blue_set.append(i)
            blue_sum += 1 / i

        # Calculate the current discrepancy
        discrepancy = red_sum - blue_sum
        potential = abs(discrepancy)
        energy = discrepancy ** 2

        # Store history for analysis
        discrepancy_history.append(discrepancy)
        potential_history.append(potential)
        energy_history.append(energy)

        # Print progress at key milestones
        if i % (n // 10) == 0:
            print(f"Step {i}/{n} - Current Discrepancy: {discrepancy}, Potential: {potential}")

    return discrepancy_history, potential_history, energy_history, red_set, blue_set

# Plotting the results to analyze potential and energy
def plot_results(n, discrepancy_history, potential_history, energy_history):
    steps = np.arange(2, n + 1)

    plt.figure(figsize=(14, 8))

    # Plot Discrepancy over time
    plt.subplot(3, 1, 1)
    plt.plot(steps, discrepancy_history, label="Discrepancy (D_n)", color='blue')
    plt.xlabel('Step (n)')
    plt.ylabel('Discrepancy (D_n)')
    plt.title('Discrepancy Evolution Over Time')
    plt.grid(True)
    plt.legend()

    # Plot Potential Function over time
    plt.subplot(3, 1, 2)
    plt.plot(steps, potential_history, label="Potential Function (|D_n|)", color='red')
    plt.xlabel('Step (n)')
    plt.ylabel('|D_n| (Potential)')
    plt.title('Potential Function Evolution')
    plt.grid(True)
    plt.legend()

    # Plot Energy Function over time
    plt.subplot(3, 1, 3)
    plt.plot(steps, energy_history, label="Energy Function (D_n^2)", color='green')
    plt.xlabel('Step (n)')
    plt.ylabel('Energy (D_n^2)')
    plt.title('Energy Function Evolution Over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to run the experiment
def main():
    # Parameters for testing
    n = 1000000  # Number of steps to test the deterministic greedy strategy

    # Run the greedy coloring strategy
    start_time = time.time()
    discrepancy_history, potential_history, energy_history, red_set, blue_set = greedy_coloring(n)
    execution_time = time.time() - start_time

    # Final discrepancy and execution time
    final_discrepancy = discrepancy_history[-1]
    print(f"\nCompleted n={n} - Final Discrepancy: {final_discrepancy:.6f}, Execution Time: {execution_time:.2f} seconds")

    # Plotting the results
    plot_results(n, discrepancy_history, potential_history, energy_history)

if __name__ == "__main__":
    main()
