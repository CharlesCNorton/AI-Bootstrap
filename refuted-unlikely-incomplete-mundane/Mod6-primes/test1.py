import cupy as cp
import numpy as np
from math import exp

def generate_primes_gpu(limit):
    is_prime = cp.ones(limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(cp.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = cp.arange(limit)[is_prime]
    return cp.asnumpy(primes)

def predict_clusters(range_size, cluster_size):
    base_freq = 0.038 * range_size
    return base_freq * exp(-0.84 * cluster_size)

def actual_clusters(start, end, size):
    primes = generate_primes_gpu(end)
    primes = primes[primes >= start]
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

    return sum(1 for c in clusters if c == size)

# Test ranges
range_sizes = [1000000, 2000000, 4000000, 8000000, 9000000, 10000000]
cluster_sizes = range(2, 8)

print("Testing complete formula:")
for m in range_sizes:
    print(f"\nRange size: {m:,}")
    print("Size | Predicted | Actual | Error%")
    print("-" * 35)

    start = 774000000
    end = start + m

    for k in cluster_sizes:
        predicted = int(predict_clusters(m, k))
        actual = actual_clusters(start, end, k)
        error = abs(predicted - actual) / actual * 100

        print(f"{k:4d} | {predicted:9d} | {actual:6d} | {error:6.2f}%")
