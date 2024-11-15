import cupy as cp
import numpy as np
from math import exp, log
from collections import defaultdict

def generate_primes_gpu(limit):
    is_prime = cp.ones(limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(cp.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = cp.arange(limit)[is_prime]
    return cp.asnumpy(primes)

def analyze_cluster_patterns(primes):
    mod6_values = primes % 6

    # Track detailed cluster information
    clusters = []
    cluster_positions = defaultdict(list)  # Store positions of each cluster size
    current_mod = mod6_values[0]
    cluster_size = 1
    cluster_start = 0

    for i in range(1, len(mod6_values)):
        if mod6_values[i] == current_mod:
            cluster_size += 1
        else:
            if cluster_size > 1:
                clusters.append((cluster_size, cluster_start, primes[cluster_start]))
                cluster_positions[cluster_size].append(primes[cluster_start])
            current_mod = mod6_values[i]
            cluster_size = 1
            cluster_start = i

    # Analyze the ~27,700 gap pattern
    print("\n1. Gap Pattern Analysis:")
    for size in range(2, 8):
        if len(cluster_positions[size]) > 1:
            gaps = np.diff(cluster_positions[size])
            print(f"\nSize {size} clusters:")
            print(f"Mean gap: {np.mean(gaps):.1f}")
            print(f"Gap StdDev: {np.std(gaps):.1f}")
            print(f"Most common gaps:", sorted([(g, count) for g, count in zip(*np.unique(gaps, return_counts=True))])[:5])

    # Analyze variability increase
    print("\n2. Variability Analysis:")
    for size in range(2, 8):
        if len(cluster_positions[size]) > 1:
            positions = np.array(cluster_positions[size])
            local_density = []
            window_size = 1000000
            for i in range(0, len(positions)-10, 10):
                window = positions[i:i+10]
                local_density.append(len(window)/(window[-1]-window[0]))

            print(f"\nSize {size} clusters:")
            print(f"Local density variation: {np.std(local_density)/np.mean(local_density)*100:.2f}%")
            print(f"Number of clusters: {len(positions)}")

    # Analyze size 2 cluster structure
    print("\n3. Size 2 Cluster Analysis:")
    size2_clusters = [c for c in clusters if c[0] == 2]

    # Look at mod 6 patterns before and after size 2 clusters
    before_after_patterns = defaultdict(int)
    for i, cluster in enumerate(size2_clusters[:-1]):
        if i > 0:
            pattern = (
                primes[cluster[1]-1] % 6 if cluster[1] > 0 else None,
                primes[cluster[1]+2] % 6 if cluster[1]+2 < len(primes) else None
            )
            before_after_patterns[pattern] += 1

    print("\nMost common patterns around size 2 clusters:")
    for pattern, count in sorted(before_after_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Before: {pattern[0]}, After: {pattern[1]}, Count: {count}")

# Generate primes and analyze
start_point = 100000000
end_point = start_point + 10000000
all_primes = generate_primes_gpu(end_point)
all_primes = all_primes[all_primes >= start_point]

analyze_cluster_patterns(all_primes)
