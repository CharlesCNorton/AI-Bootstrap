import cupy as cp
import numpy as np

def generate_primes_segmented(limit, segment_size=int(1e8)):
    """Generates primes up to `limit` using a segmented Sieve of Eratosthenes on the GPU."""
    print(f"[INFO] Starting segmented prime generation up to {limit} with segment size {segment_size}...")
    primes = cp.array([2, 3], dtype=cp.int64)  # Start with known small primes for initialization
    segment_count = 0  # Counter for tracking segment progress

    for low in range(5, limit + 1, segment_size):
        high = min(low + segment_size - 1, limit)
        print(f"[INFO] Processing segment: {low} to {high}...")

        # Initialize segment as prime (True)
        sieve = cp.ones(high - low + 1, dtype=bool)
        print(f"[INFO] Segment initialized. Running sieve for known primes up to sqrt({limit})...")

        # Cross out multiples of known primes from previous segments
        for prime in primes:
            if prime * prime > high:
                print(f"[INFO] Prime {prime} is beyond segment range {high}. Stopping loop.")
                break
            start = max(prime * prime, low + (prime - low % prime) % prime)
            sieve[start - low::prime] = False
            print(f"[INFO] Marked multiples of prime {prime} starting from {start}.")

        # Identify new primes in the current segment
        segment_primes = cp.nonzero(sieve)[0] + low
        print(f"[INFO] Segment {segment_count} found {len(segment_primes)} primes.")

        # Append new primes to the primes list
        primes = cp.concatenate((primes, segment_primes))
        segment_count += 1
        print(f"[INFO] Completed segment {segment_count}. Total primes found so far: {len(primes)}.")

    print("[INFO] Prime generation complete.")
    return primes

def count_prime_clusters_gpu(primes, modulo_base=6):
    """Counts size-2 prime clusters where consecutive primes have the same residue modulo `modulo_base` on the GPU."""
    print("[INFO] Starting GPU-based cluster counting...")

    # Calculate residues modulo the given base
    print("[INFO] Calculating residues...")
    residues = primes % modulo_base
    print("[INFO] Residues calculated. Starting cluster comparison...")

    # Identify consecutive primes with the same residue
    same_residue = residues[:-1] == residues[1:]
    clusters = cp.sum(same_residue)

    print(f"[INFO] Cluster counting complete. Total clusters found: {clusters.get()}")
    return clusters.get()  # Move result to CPU memory

# Configuration for extreme range testing
limit = int(1e12)  # Upper limit for prime generation
segment_size = int(1e8)  # Segment size for the sieve
print(f"[INFO] Generating primes up to {limit} with segment size {segment_size}...")

# Step 1: Generate primes up to the limit using the segmented sieve approach
primes = generate_primes_segmented(limit, segment_size=segment_size)
print("[INFO] Prime generation complete. Moving to cluster counting phase...")

# Step 2: Count size-2 prime clusters modulo 6
print("[INFO] Starting to count size-2 prime clusters modulo 6...")
clusters = count_prime_clusters_gpu(primes)
print(f"[RESULT] Total size-2 prime clusters modulo 6: {clusters}")
