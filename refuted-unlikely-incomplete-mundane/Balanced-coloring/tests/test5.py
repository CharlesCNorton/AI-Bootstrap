import random
import time
import gc
import heapq
from decimal import Decimal, getcontext

# Set precision for Decimal computations
getcontext().prec = 50  # Adjust precision as needed (e.g., 50 decimal places)

# Garbage collection after each test
def clean_up():
    gc.collect()

# Greedy Algorithm with Decimal
def greedy_algorithm(n):
    red_sum = Decimal(0)
    blue_sum = Decimal(0)
    for i in range(2, n + 2):
        reciprocal = Decimal(1) / Decimal(i)
        if red_sum < blue_sum:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return abs(red_sum - blue_sum)

# Weighted Averaging Algorithm (50/50 split) with Decimal
def weighted_averaging_algorithm(n):
    red_sum = Decimal(0)
    blue_sum = Decimal(0)
    for i in range(2, n + 2):
        reciprocal = Decimal(1) / Decimal(i)
        if random.random() < 0.5:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return abs(red_sum - blue_sum)

# Karmarkar-Karp Algorithm using a max-heap with Decimal
def karmarkar_karp_algorithm(n):
    # Create a list of Decimals for reciprocals
    heap = [Decimal(1) / Decimal(i) for i in range(2, n + 2)]
    # Convert to a max-heap using negative values
    heap = [-x for x in heap]
    heapq.heapify(heap)
    while len(heap) > 1:
        largest = -heapq.heappop(heap)
        second_largest = -heapq.heappop(heap)
        new_number = largest - second_largest
        heapq.heappush(heap, -new_number)
    return abs(heap[0])

# Run comparison for a given size and algorithm set
def run_comparison(n):
    print(f"\nTesting with n = {n} integers")

    # Greedy Algorithm
    start_time = time.time()
    greedy_result = greedy_algorithm(n)
    greedy_time = time.time() - start_time
    print(f"Greedy Algorithm: Discrepancy = {greedy_result}, Time = {greedy_time:.2f} seconds")

    clean_up()

    # Weighted Averaging Algorithm
    start_time = time.time()
    wa_result = weighted_averaging_algorithm(n)
    wa_time = time.time() - start_time
    print(f"Weighted Averaging Algorithm: Discrepancy = {wa_result}, Time = {wa_time:.2f} seconds")

    clean_up()

    # Karmarkar-Karp Algorithm
    start_time = time.time()
    kk_result = karmarkar_karp_algorithm(n)
    kk_time = time.time() - start_time
    print(f"Karmarkar-Karp Algorithm: Discrepancy = {kk_result}, Time = {kk_time:.2f} seconds")

    clean_up()

# Example usage
# Run comparisons for different sizes
n_values = [100000, 1000000, 10000000]  # Adjust n as needed
for n in n_values:
    run_comparison(n)
