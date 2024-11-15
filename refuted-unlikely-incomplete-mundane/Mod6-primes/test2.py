import cupy as cp
import numpy as np
from math import exp, log

def generate_primes_gpu(limit):
    is_prime = cp.ones(limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(cp.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = cp.arange(limit)[is_prime]
    return cp.asnumpy(primes)

def count_clusters_in_strict_range(start, end):
    # Generate primes
    primes = generate_primes_gpu(end + 1)  # +1 to ensure we get end if it's prime

    # Create boolean mask for our range
    in_range = (primes >= start) & (primes <= end)
    range_primes = primes[in_range]

    # Debug output
    print(f"\nRange: {start:,} to {end:,}")
    print(f"Number of primes in range: {len(range_primes)}")
    if len(range_primes) > 0:
        print(f"First prime: {range_primes[0]}")
        print(f"Last prime: {range_primes[-1]}")

    # Get mod 6 values
    mod6_values = range_primes % 6

    # Count clusters
    clusters = []
    if len(mod6_values) > 0:
        current_mod = mod6_values[0]
        cluster_size = 1

        for i in range(1, len(mod6_values)):
            if mod6_values[i] == current_mod:
                cluster_size += 1
            else:
                if cluster_size > 1:
                    clusters.append(cluster_size)
                current_mod = mod6_values[i]
                cluster_size = 1

        if cluster_size > 1:
            clusters.append(cluster_size)

    # Count by size
    counts = {}
    for size in range(2, 8):
        counts[size] = sum(1 for c in clusters if c == size)

    return counts

# Test with non-overlapping ranges
start_point = 100000000
range_size = 1000000

print("Testing with non-overlapping ranges:")
for i in range(5):
    start = start_point + (i * range_size)
    end = start + range_size - 1

    counts = count_clusters_in_strict_range(start, end)

    print(f"\nSegment {i+1} cluster counts:")
    for size in range(2, 8):
        print(f"Size {size}: {counts[size]}")
