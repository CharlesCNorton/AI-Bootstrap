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

# Generate primes
start_point = 100000000
end_point = start_point + 10000000
all_primes = generate_primes_gpu(end_point)
all_primes = all_primes[all_primes >= start_point]

# Verify the size 2 cluster theory
mod6_values = all_primes % 6

# Track violations and patterns
violations = 0
total_clusters = 0
pattern_counts = defaultdict(int)
gap_patterns = defaultdict(int)

current_mod = mod6_values[0]
cluster_start = 0
cluster_size = 1

print("Analyzing size 2 cluster structure...")
print("\nChecking for pattern violations and collecting statistics...")

for i in range(1, len(mod6_values)):
    if mod6_values[i] == current_mod:
        cluster_size += 1
        if cluster_size == 2:
            total_clusters += 1
            if cluster_start > 0 and i+1 < len(mod6_values):
                # Record the pattern
                pattern = (mod6_values[cluster_start-1], current_mod, mod6_values[i+1])
                pattern_counts[pattern] += 1

                # Check gaps
                prev_gap = all_primes[cluster_start] - all_primes[cluster_start-1]
                next_gap = all_primes[i+1] - all_primes[i]
                gap_pattern = (prev_gap % 6, next_gap % 6)
                gap_patterns[gap_pattern] += 1

                # Check for violations of our theory
                if prev_gap % 6 not in [2, 4]:
                    violations += 1
    else:
        cluster_size = 1
        cluster_start = i
        current_mod = mod6_values[i]

print("\nResults:")
print(f"Total size 2 clusters examined: {total_clusters}")
print(f"Theory violations: {violations}")
print(f"Theory accuracy: {100 - (violations/total_clusters*100):.2f}%")

print("\nMost common mod 6 patterns (prev, cluster, next):")
for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Pattern {pattern}: {count} occurrences ({count/total_clusters*100:.1f}%)")

print("\nMost common gap patterns (prev mod 6, next mod 6):")
for gap_pattern, count in sorted(gap_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Gaps {gap_pattern}: {count} occurrences ({count/total_clusters*100:.1f}%)")

# Additional verification of our key predictions
print("\nVerifying key predictions:")
mod1_to_5 = sum(1 for p in pattern_counts.items() if p[0][1:] == (5,1))
mod5_to_1 = sum(1 for p in pattern_counts.items() if p[0][1:] == (1,5))
print(f"1→5→1 vs 5→1→5 ratio: {mod1_to_5/mod5_to_1:.3f}")

gap2_count = sum(count for (prev,_),count in gap_patterns.items() if prev % 6 == 2)
gap4_count = sum(count for (prev,_),count in gap_patterns.items() if prev % 6 == 4)
print(f"Gap ≡ 2 vs Gap ≡ 4 ratio: {gap2_count/gap4_count:.3f}")
