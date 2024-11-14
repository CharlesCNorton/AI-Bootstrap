import cupy as cp
import numpy as np
import time

def generate_primes_in_range(limit_low, limit_high):
    """Generates primes within a specified range using GPU-accelerated Sieve of Eratosthenes."""
    range_size = limit_high - limit_low
    is_prime = cp.ones(range_size, dtype=bool)
    is_prime[0:2] = False  # Start with numbers 0, 1 not prime

    print(f"[{time.strftime('%H:%M:%S')}] Generating primes in range {limit_low} to {limit_high} using GPU...")

    start_time = time.time()
    for i in range(2, int(cp.sqrt(limit_high)) + 1):
        if i >= limit_low:
            start_index = i * 2 - limit_low  # Direct multiples within the range
        else:
            start_index = ((limit_low + i - 1) // i) * i - limit_low

        if start_index < range_size:
            is_prime[start_index::i] = False

    primes = cp.nonzero(is_prime)[0] + limit_low  # Convert to absolute indices
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Prime generation complete. {len(primes)} primes found (Time: {elapsed:.2f} sec)")
    return primes

def count_prime_clusters(primes, modulo_base=6):
    """Counts consecutive size-2 prime clusters with the same residue modulo a given base."""
    residues = primes % modulo_base
    clusters = cp.sum(residues[1:] == residues[:-1])  # Count consecutive matching residues
    clusters = clusters.get()  # Convert to numpy-compatible format
    print(f"[{time.strftime('%H:%M:%S')}] Counted {clusters} size-2 prime clusters modulo {modulo_base}.")
    return clusters

# Configuration for real sampling
target_range_start, target_range_end = int(1e10), int(1e10 + 1e9)
sample_count = 50  # Number of subranges to sample within the target range
subrange_size = int(1e6)  # Size of each sampled subrange

# Progress tracking setup
cluster_counts = []
print(f"Starting real sampling computation across target range {target_range_start} to {target_range_end}.\n")
overall_start_time = time.time()

for i in range(sample_count):
    subrange_start = target_range_start + i * subrange_size * 20  # Spread subranges evenly
    subrange_end = subrange_start + subrange_size
    percent_complete = (i + 1) / sample_count * 100

    print(f"--- Sample {i+1}/{sample_count} ({percent_complete:.2f}% Complete) ---")
    print(f"[{time.strftime('%H:%M:%S')}] Subrange: {subrange_start} to {subrange_end}")

    sample_start_time = time.time()
    primes = generate_primes_in_range(subrange_start, subrange_end)
    clusters = count_prime_clusters(primes)
    cluster_counts.append(clusters)

    sample_elapsed = time.time() - sample_start_time
    print(f"[{time.strftime('%H:%M:%S')}] Sample {i+1} complete. Clusters found: {clusters} (Sample Time: {sample_elapsed:.2f} sec)\n")

# Transfer `cluster_counts` to numpy-compatible array
cluster_counts_np = np.array(cluster_counts)

overall_elapsed = time.time() - overall_start_time
average_density = np.mean(cluster_counts_np) / subrange_size
total_clusters_estimate = average_density * (target_range_end - target_range_start)

# Final report with timing and results
real_sampling_results = {
    "Total computation time (minutes)": overall_elapsed / 60,
    "Average cluster density per sampled subrange": average_density,
    "Estimated total size-2 prime clusters modulo 6": total_clusters_estimate
}

real_sampling_results
