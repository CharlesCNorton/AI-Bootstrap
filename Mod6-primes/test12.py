import numpy as np
from math import sqrt, log
from scipy.optimize import curve_fit

def segmented_sieve_range(start, end):
    """
    Generate primes in the range [start, end) using a segmented sieve.
    """
    limit = end
    segment_size = 10_000_000  # Adjust based on memory capacity
    sqrt_limit = int(sqrt(limit)) + 1
    primes = simple_sieve(sqrt_limit)
    low = max(start, sqrt_limit)
    high = low + segment_size

    # Initialize sieve for [start, min(sqrt_limit, end))
    if start < sqrt_limit:
        sieve = np.ones(sqrt_limit - start, dtype=bool)
        sieve[:2 - start] = False if start <= 2 else True
        for p in range(2, int(sqrt(sqrt_limit)) + 1):
            if sieve[p - start]:
                sieve[p*p - start::p] = False
        segment_primes = np.nonzero(sieve)[0] + start
        yield segment_primes

    # Sieve segments from sqrt_limit to end
    while low < limit:
        if high > limit:
            high = limit
        sieve = np.ones(high - low, dtype=bool)
        for p in primes:
            start_index = (-low) % p
            sieve[start_index::p] = False
        segment_primes = np.nonzero(sieve)[0] + low
        yield segment_primes
        low = high
        high += segment_size

def simple_sieve(limit):
    """
    Generate all primes up to 'limit' using the basic sieve.
    """
    sieve = np.ones(limit, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.nonzero(sieve)[0]

def count_clusters_in_range(start, end):
    """
    Count size-2 clusters modulo 6 in the range [start, end).
    """
    total_clusters = 0
    prev_mod6 = -1  # Initialize to invalid value
    cluster_size = 0

    for segment_primes in segmented_sieve_range(start, end):
        mod6_values = segment_primes % 6
        if len(mod6_values) == 0:
            continue
        for mod in mod6_values:
            if mod == prev_mod6:
                cluster_size += 1
                if cluster_size == 2:
                    total_clusters += 1
            else:
                cluster_size = 1
                prev_mod6 = mod
    return total_clusters

def prediction_formula(s, r, a, b, c):
    log_term = np.log(s / 1e8)
    denominator = 1 + b * log_term + c * (log_term ** 2)
    return r * a / denominator

# Define the subranges
start_point = 0
end_point = 2_000_000_000
range_size = 100_000_000  # Adjust based on computational resources

range_starts = np.arange(start_point, end_point, range_size)
range_ends = range_starts + range_size
range_ends[-1] = min(range_ends[-1], end_point)  # Ensure the last end point does not exceed the limit

actual_cluster_counts = []
s_values = []
r_values = []

print("Generating primes and counting actual clusters in multiple ranges...")
for start, end in zip(range_starts, range_ends):
    print(f"\nProcessing range {start:,} to {end:,}...")
    total_clusters = count_clusters_in_range(start, end)
    actual_cluster_counts.append(total_clusters)
    s_values.append(start)
    r_values.append(end - start)
    print(f"Clusters found: {total_clusters}")

s_values = np.array(s_values)
actual_cluster_counts = np.array(actual_cluster_counts)
r_values = np.array(r_values)

# Initial guesses for a, b, c
initial_a = 0.0138226  # Based on previous base_rate
initial_b = 0.05       # Based on previous decay_rate
initial_c = 0.0        # Start with zero for the quadratic term

# Perform curve fitting with multiple data points
print("\nOptimizing constants...")
popt, pcov = curve_fit(
    f=lambda s, a, b, c: prediction_formula(s, r_values, a, b, c),
    xdata=s_values,
    ydata=actual_cluster_counts,
    p0=[initial_a, initial_b, initial_c],
    bounds=(0, np.inf)
)

optimized_a, optimized_b, optimized_c = popt

# Calculate predicted clusters with optimized constants
predicted_clusters = prediction_formula(s_values, r_values, optimized_a, optimized_b, optimized_c)
errors_percentage = abs(predicted_clusters - actual_cluster_counts) / actual_cluster_counts * 100

# Output the results
print("\nOptimization Results:")
print(f"Optimized a (base_rate): {optimized_a}")
print(f"Optimized b (linear decay): {optimized_b}")
print(f"Optimized c (quadratic decay): {optimized_c}\n")

print("Range-wise Predictions:")
for i, s in enumerate(s_values):
    print(f"Range {s:,} to {s + r_values[i]:,}:")
    print(f"Predicted clusters: {int(predicted_clusters[i])}")
    print(f"Actual clusters: {actual_cluster_counts[i]}")
    print(f"Error percentage: {errors_percentage[i]:.2f}%\n")
