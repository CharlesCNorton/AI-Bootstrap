import cupy as cp
import numpy as np
from collections import defaultdict

def generate_primes_gpu(limit):
    is_prime = cp.ones(limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(cp.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = cp.arange(limit)[is_prime]
    return cp.asnumpy(primes)

def analyze_size2_structure(primes):
    mod6_values = primes % 6

    # Track detailed size 2 cluster information
    clusters = []
    current_mod = mod6_values[0]
    cluster_size = 1
    cluster_start = 0

    # For each size 2 cluster, record:
    # 1. The gap leading to it
    # 2. The gap following it
    # 3. The mod 6 values of surrounding primes
    cluster_context = []

    for i in range(1, len(mod6_values)):
        if mod6_values[i] == current_mod:
            cluster_size += 1
            if cluster_size == 2:
                if cluster_start > 0 and i+1 < len(primes):
                    prev_gap = primes[cluster_start] - primes[cluster_start-1]
                    next_gap = primes[i+1] - primes[i]
                    cluster_context.append({
                        'prev_gap': prev_gap,
                        'next_gap': next_gap,
                        'prev_mod': mod6_values[cluster_start-1],
                        'cluster_mod': current_mod,
                        'next_mod': mod6_values[i+1]
                    })
        else:
            cluster_size = 1
            cluster_start = i
            current_mod = mod6_values[i]

    print("\nDetailed Size 2 Cluster Analysis:")

    # Analyze gap patterns
    print("\nGap Combinations (prev_gap, next_gap):")
    gap_pairs = defaultdict(int)
    for ctx in cluster_context:
        gap_pairs[(ctx['prev_gap'], ctx['next_gap'])] += 1

    for (prev, next), count in sorted(gap_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Prev gap: {prev}, Next gap: {next}, Count: {count}")

    # Analyze mod 6 transitions
    print("\nMod 6 Transition Patterns:")
    mod_patterns = defaultdict(int)
    for ctx in cluster_context:
        pattern = (ctx['prev_mod'], ctx['cluster_mod'], ctx['next_mod'])
        mod_patterns[pattern] += 1

    for pattern, count in sorted(mod_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Pattern {pattern}: {count} occurrences")

    # Additional analysis: Look at the distribution of gaps mod 6
    print("\nGap distributions mod 6:")
    prev_gaps_mod6 = defaultdict(int)
    next_gaps_mod6 = defaultdict(int)
    for ctx in cluster_context:
        prev_gaps_mod6[ctx['prev_gap'] % 6] += 1
        next_gaps_mod6[ctx['next_gap'] % 6] += 1

    print("\nPrevious gaps mod 6:")
    for gap, count in sorted(prev_gaps_mod6.items()):
        print(f"Gap ≡ {gap} (mod 6): {count} occurrences")

    print("\nNext gaps mod 6:")
    for gap, count in sorted(next_gaps_mod6.items()):
        print(f"Gap ≡ {gap} (mod 6): {count} occurrences")

# Generate primes and analyze
start_point = 100000000
end_point = start_point + 10000000
all_primes = generate_primes_gpu(end_point)
all_primes = all_primes[all_primes >= start_point]

analyze_size2_structure(all_primes)
