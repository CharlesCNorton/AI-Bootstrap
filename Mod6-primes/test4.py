import cupy as cp
import numpy as np
from math import exp, log
from statistics import mean, stdev

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
    return counts, clusters  # Return both counts and raw clusters

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

print("Extended Window Analysis:")
window_size = 2000000
step_size = 500000  # Smaller steps for more overlap
num_steps = 15      # More windows

# Store results for statistical analysis
window_results = {k: [] for k in range(2, 8)}
window_errors = {k: [] for k in range(2, 8)}
cluster_gaps = {k: [] for k in range(2, 8)}

for i in range(num_steps):
    window_start = i * step_size
    window_end = window_start + window_size

    window_start_idx = np.searchsorted(all_primes, all_primes[0] + window_start)
    window_end_idx = np.searchsorted(all_primes, all_primes[0] + window_end)
    window_primes = all_primes[window_start_idx:window_end_idx]

    print(f"\nWindow {i+1}:")
    print(f"Range: {start_point + window_start:,} to {start_point + window_start + window_size:,}")
    print("Size | Predicted | Actual | Error% | Avg Gap")
    print("-" * 45)

    actual_counts, raw_clusters = count_clusters_in_range(window_primes)

    # Calculate gaps between clusters of same size
    for cluster in raw_clusters:
        if cluster < 8:
            cluster_gaps[cluster].append(len(raw_clusters))

    for k in range(2, 8):
        predicted = int(predict_clusters_final(window_size, k, start_point + window_start))
        actual = actual_counts[k]
        error = abs(predicted - actual) / actual * 100

        window_results[k].append(actual)
        window_errors[k].append(error)

        avg_gap = mean(cluster_gaps[k]) if cluster_gaps[k] else 0
        print(f"{k:4d} | {predicted:9d} | {actual:6d} | {error:6.2f}% | {avg_gap:6.1f}")

print("\nStatistical Analysis:")
for k in range(2, 8):
    counts = window_results[k]
    errors = window_errors[k]
    print(f"\nCluster Size {k}:")
    print(f"Count Statistics:")
    print(f"  Mean: {mean(counts):.1f}")
    print(f"  StdDev: {stdev(counts):.1f}")
    print(f"  Coefficient of Variation: {stdev(counts)/mean(counts)*100:.2f}%")
    print(f"Error Statistics:")
    print(f"  Mean Error: {mean(errors):.2f}%")
    print(f"  StdDev of Error: {stdev(errors):.2f}%")
    if cluster_gaps[k]:
        print(f"Gap Statistics:")
        print(f"  Mean Gap: {mean(cluster_gaps[k]):.1f}")
        print(f"  StdDev of Gap: {stdev(cluster_gaps[k]):.1f}")
