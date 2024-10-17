import numpy as np
import time
import heapq
import statistics
import gc

# Greedy Algorithm
def greedy_algorithm(n):
    print(f"\nStarting Greedy Algorithm for n = {n}")
    red_sum = 0.0
    blue_sum = 0.0
    discrepancies = []

    for i in range(2, n + 2):
        reciprocal = 1 / i
        # Assign reciprocal to the subset with the smaller current sum
        if red_sum < blue_sum:
            red_sum += reciprocal
            assigned_set = "Red"
        else:
            blue_sum += reciprocal
            assigned_set = "Blue"

        discrepancy = abs(red_sum - blue_sum)
        discrepancies.append(discrepancy)

        # Print intermediate status every 10^5 iterations or at the last step
        if i % 100000 == 0 or i == n + 1:
            print(f"Iteration {i}: Assigned 1/{i} to {assigned_set}. Current Discrepancy = {discrepancy:.6e}")

    print(f"Completed Greedy Algorithm for n = {n}. Final Discrepancy = {discrepancies[-1]:.6e}")
    return discrepancies[-1]

# Karmarkar-Karp Algorithm
def karmarkar_karp_algorithm(n):
    print(f"\nStarting Karmarkar-Karp Algorithm for n = {n}")
    reciprocals = [1 / i for i in range(2, n + 2)]
    # Use a max heap to combine the largest elements
    max_heap = [-x for x in reciprocals]
    heapq.heapify(max_heap)

    iteration = 0
    while len(max_heap) > 1:
        first = -heapq.heappop(max_heap)
        second = -heapq.heappop(max_heap)
        # Combine the two largest elements and push their difference back
        heapq.heappush(max_heap, -(first - second))
        iteration += 1

        # Print status every 10^5 iterations
        if iteration % 100000 == 0 or len(max_heap) == 1:
            print(f"Iteration {iteration}: Combined {first:.6e} and {second:.6e}. Remaining elements = {len(max_heap)}")

    discrepancy = -max_heap[0] if max_heap else 0.0
    print(f"Completed Karmarkar-Karp Algorithm for n = {n}. Final Discrepancy = {discrepancy:.6e}")
    return discrepancy

# Run experiments sequentially and generate statistical summaries
def run_experiments():
    n_values = [10**i for i in range(5, 10)]  # n = 10^5 to 10^9
    greedy_discrepancies = []
    kk_discrepancies = []
    greedy_times = []
    kk_times = []

    for n in n_values:
        print(f"\nRunning experiments for n = {n}")

        # Greedy Algorithm
        print("\n--- Running Greedy Algorithm ---")
        start_time = time.time()
        greedy_discrepancy = greedy_algorithm(n)
        end_time = time.time()
        greedy_time = end_time - start_time
        greedy_discrepancies.append(greedy_discrepancy)
        greedy_times.append(greedy_time)
        print(f"Greedy Algorithm for n = {n} took {greedy_time:.2f} seconds")

        # Garbage collect to free memory
        gc.collect()
        print("Memory cleared after Greedy Algorithm")

        # Karmarkar-Karp Algorithm
        print("\n--- Running Karmarkar-Karp Algorithm ---")
        start_time = time.time()
        kk_discrepancy = karmarkar_karp_algorithm(n)
        end_time = time.time()
        kk_time = end_time - start_time
        kk_discrepancies.append(kk_discrepancy)
        kk_times.append(kk_time)
        print(f"Karmarkar-Karp Algorithm for n = {n} took {kk_time:.2f} seconds")

        # Garbage collect to free memory
        gc.collect()
        print("Memory cleared after Karmarkar-Karp Algorithm")

    # Generate statistical summaries
    generate_statistics(n_values, greedy_discrepancies, kk_discrepancies, greedy_times, kk_times)

# Generate statistical summaries
def generate_statistics(n_values, greedy_discrepancies, kk_discrepancies, greedy_times, kk_times):
    def calculate_statistics(data, label):
        print(f"\n{label} Statistics:")
        print(f"  Mean: {statistics.mean(data):.6e}")
        print(f"  Median: {statistics.median(data):.6e}")
        print(f"  Standard Deviation: {statistics.stdev(data):.6e}" if len(data) > 1 else "  Standard Deviation: N/A")
        print(f"  Min: {min(data):.6e}")
        print(f"  Max: {max(data):.6e}")

    # Print statistics for Greedy Algorithm
    calculate_statistics(greedy_discrepancies, "Greedy Algorithm Discrepancies")
    calculate_statistics(greedy_times, "Greedy Algorithm Execution Times")

    # Print statistics for Karmarkar-Karp Algorithm
    calculate_statistics(kk_discrepancies, "Karmarkar-Karp Algorithm Discrepancies")
    calculate_statistics(kk_times, "Karmarkar-Karp Algorithm Execution Times")

    # Print comparison summary
    print("\nComparison Summary (Greedy vs Karmarkar-Karp):")
    for i, n in enumerate(n_values):
        print(f"n = {n}:")
        print(f"  Greedy Discrepancy: {greedy_discrepancies[i]:.6e}, Time: {greedy_times[i]:.2f} s")
        print(f"  KK Discrepancy: {kk_discrepancies[i]:.6e}, Time: {kk_times[i]:.2f} s")
        print("-" * 60)

# Run the experiment
if __name__ == "__main__":
    print("Starting the experiment...")
    run_experiments()
    print("Experiment completed.")
