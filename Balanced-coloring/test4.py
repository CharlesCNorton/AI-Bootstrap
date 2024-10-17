import random
import time
import gc
import heapq

# Garbage collection after each test
def clean_up():
    gc.collect()

# Greedy Algorithm
def greedy_algorithm(n):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if red_sum < blue_sum:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return abs(red_sum - blue_sum)

# Weighted Averaging Algorithm (50/50 split)
def weighted_averaging_algorithm(n):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if random.random() < 0.5:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return abs(red_sum - blue_sum)

# Karmarkar-Karp (KK) Algorithm using a max-heap
def karmarkar_karp_algorithm(n):
    heap = []
    for i in range(2, n + 2):
        heapq.heappush(heap, -1 / i)  # Use negative for max heap behavior
    while len(heap) > 1:
        largest = -heapq.heappop(heap)
        second_largest = -heapq.heappop(heap)
        heapq.heappush(heap, -(largest - second_largest))
    return abs(heap[0])

# Run comparison for a given size and algorithm set
def run_comparison(n):
    print(f"\nTesting with n = {n} integers")

    # Greedy
    start_time = time.time()
    greedy_result = greedy_algorithm(n)
    greedy_time = time.time() - start_time
    print(f"Greedy Algorithm: Discrepancy = {greedy_result:.10e}, Time = {greedy_time:.2f} seconds")

    clean_up()

    # Weighted Averaging
    start_time = time.time()
    wa_result = weighted_averaging_algorithm(n)
    wa_time = time.time() - start_time
    print(f"Weighted Averaging Algorithm: Discrepancy = {wa_result:.10e}, Time = {wa_time:.2f} seconds")

    clean_up()

    # Karmarkar-Karp
    start_time = time.time()
    kk_result = karmarkar_karp_algorithm(n)
    kk_time = time.time() - start_time
    print(f"Karmarkar-Karp Algorithm: Discrepancy = {kk_result:.10e}, Time = {kk_time:.2f} seconds")

    clean_up()

# Run comparisons for different sizes
for size in [10**5, 10**6, 10**7, 10**8, 10**9]:
    run_comparison(size)
