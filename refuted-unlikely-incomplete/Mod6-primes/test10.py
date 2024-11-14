import cupy as cp
import numpy as np
from math import log

def generate_primes_gpu(limit):
    is_prime = cp.ones(limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(cp.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = cp.arange(limit)[is_prime]
    return cp.asnumpy(primes)

ranges = [
    (100_000_000, 110_000_000),
    (200_000_000, 210_000_000),
    (500_000_000, 510_000_000),
    (1_000_000_000, 1_010_000_000),
    (1_010_000_000, 2_000_000_000)
]

base_rate = 0.0138226
for start, end in ranges:
    primes = generate_primes_gpu(end)
    primes = primes[primes >= start]
    mod6_values = primes % 6

    total_clusters = 0
    current_mod = mod6_values[0]
    cluster_size = 1

    for i in range(1, len(mod6_values)):
        if mod6_values[i] == current_mod:
            cluster_size += 1
            if cluster_size == 2:
                total_clusters += 1
        else:
            cluster_size = 1
            current_mod = mod6_values[i]

    range_size = end - start
    correction = 1 / (1 + 0.05 * log(start/1e8))
    predicted_clusters = int(range_size * base_rate * correction)

    print(f"\nRange {start:,} to {end:,}:")
    print(f"Actual clusters: {total_clusters}")
    print(f"Predicted clusters: {predicted_clusters}")
    print(f"Error: {abs(predicted_clusters-total_clusters)/total_clusters*100:.2f}%")
