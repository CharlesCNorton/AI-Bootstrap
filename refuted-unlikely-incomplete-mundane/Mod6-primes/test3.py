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

def count_clusters_in_range(primes):
    mod6_values = primes % 6

    clusters = []
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

    counts = {k: sum(1 for c in clusters if c == k) for k in range(2, 8)}
    return counts

def predict_clusters_final(range_size, cluster_size, start_point):
    log_start = log(start_point)
    base_freq = 0.049 - 0.003 * (log_start - log(1e8))
    decay_rate = 0.84 + 0.03 * exp(-(log_start - log(1e8))/2)
    scaling = 0.9
    return scaling * base_freq * range_size * exp(-decay_rate * cluster_size)

# Generate primes for our full range
start_point = 100000000
end_point = start_point + 10000000
all_primes = generate_primes_gpu(end_point)
all_primes = all_primes[all_primes >= start_point]

print("1. Testing different range sizes:")
range_sizes = [1000000, 2000000, 3000000, 4000000, 5000000]
cluster_sizes = range(2, 8)

for range_size in range_sizes:
    print(f"\nRange size: {range_size:,}")
    print("Size | Predicted | Actual | Error%")
    print("-" * 35)

    range_end = np.searchsorted(all_primes, all_primes[0] + range_size)
    range_primes = all_primes[:range_end]
    actual_counts = count_clusters_in_range(range_primes)

    for k in cluster_sizes:
        predicted = int(predict_clusters_final(range_size, k, start_point))
        actual = actual_counts[k]
        error = abs(predicted - actual) / actual * 100
        print(f"{k:4d} | {predicted:9d} | {actual:6d} | {error:6.2f}%")

print("\n2. Testing overlapping ranges:")
window_size = 2000000
step_size = 1000000
num_steps = 5

for i in range(num_steps):
    window_start = i * step_size
    window_end = window_start + window_size

    # Get the actual primes for this window
    window_start_idx = np.searchsorted(all_primes, all_primes[0] + window_start)
    window_end_idx = np.searchsorted(all_primes, all_primes[0] + window_end)
    window_primes = all_primes[window_start_idx:window_end_idx]

    print(f"\nWindow {i+1}:")
    print(f"Range: {start_point + window_start:,} to {start_point + window_start + window_size:,}")
    print("Size | Predicted | Actual | Error%")
    print("-" * 35)

    actual_counts = count_clusters_in_range(window_primes)

    for k in cluster_sizes:
        predicted = int(predict_clusters_final(window_size, k, start_point + window_start))
        actual = actual_counts[k]
        error = abs(predicted - actual) / actual * 100
        print(f"{k:4d} | {predicted:9d} | {actual:6d} | {error:6.2f}%")
