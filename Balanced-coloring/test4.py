import numpy as np
import time
from math import log

# Optimized Sieve of Eratosthenes to generate the first n primes
def generate_primes_sieve(n):
    if n < 1:
        return []
    if n == 1:
        return [2]

    # Estimate upper bound for the n-th prime using Rosser's theorem
    # p_n < n (ln n + ln ln n) for n >= 1
    if n < 6:
        upper_bound = 15
    else:
        upper_bound = int(n * (log(n) + log(log(n)))) + 10

    # Initialize sieve: True indicates potential primality
    sieve = np.ones(upper_bound + 1, dtype=bool)
    sieve[:2] = False  # 0 and 1 are not primes

    sqrt_upper = int(upper_bound**0.5) + 1
    for p in range(2, sqrt_upper):
        if sieve[p]:
            sieve[p*p:upper_bound + 1:p] = False

    # Extract primes from sieve
    primes = np.nonzero(sieve)[0]

    # Check if we have enough primes
    if len(primes) >= n:
        return primes[:n].tolist()
    else:
        # If not enough primes, increase the upper bound and sieve again
        return generate_primes_sieve(n)  # Recursive call with higher upper_bound

# Greedy Coloring Strategy Function
def greedy_coloring(n, sequence_type='natural'):
    red_sum, blue_sum = 0.0, 0.0
    red_set, blue_set = [], []

    discrepancy_history = []
    energy_history = []
    potential_history = []

    # Select sequence type: Natural, odd numbers, multiples of a number, or primes
    if sequence_type == 'natural':
        sequence = range(2, n + 2)
    else:
        sequence = generate_custom_sequence(n, sequence_type)

    for idx, i in enumerate(sequence, 1):
        # Determine which choice minimizes the discrepancy
        if abs((red_sum + 1 / i) - blue_sum) < abs((blue_sum + 1 / i) - red_sum):
            red_set.append(i)
            red_sum += 1 / i
        else:
            blue_set.append(i)
            blue_sum += 1 / i

        # Calculate the current discrepancy
        discrepancy = red_sum - blue_sum
        potential = abs(discrepancy)
        energy = discrepancy ** 2

        # Store history for analysis
        discrepancy_history.append(discrepancy)
        potential_history.append(potential)
        energy_history.append(energy)

        # Print progress at key milestones including energy function
        if idx % (n // 10) == 0:
            print(f"Step {idx}/{n} [{sequence_type}] - Discrepancy: {discrepancy:.6e}, Potential: {potential:.6e}, Energy: {energy:.6e}")

    # Final result printout
    final_discrepancy = discrepancy_history[-1]
    final_potential = potential_history[-1]
    final_energy = energy_history[-1]
    print(f"\nCompleted n={n} [{sequence_type}] - Final Discrepancy: {final_discrepancy:.6e}, Final Potential: {final_potential:.6e}, Final Energy: {final_energy:.6e}")

    return discrepancy_history, potential_history, energy_history, red_set, blue_set

# Generate specific sequences (odd numbers, multiples, primes)
def generate_custom_sequence(n, sequence_type):
    if sequence_type == 'odd':
        return list(range(3, 2 * n + 3, 2))  # Odd numbers starting from 3
    elif sequence_type == 'multiples_of_3':
        return [3 * i for i in range(1, n + 1)]  # Multiples of 3
    elif sequence_type == 'primes':
        return generate_primes_sieve(n)
    else:
        raise ValueError("Unknown sequence type")

# Main function to run multiple experiments
def main():
    # Parameters for testing: multiple values of n and different sequence types
    test_values_n = [10**6, 10**7, 10**8]  # Testing for different sizes; adjust as needed for practicality
    sequence_types = ['natural', 'odd', 'multiples_of_3', 'primes']

    # Loop over different values of n and sequence types
    for n in test_values_n:
        for sequence_type in sequence_types:
            print(f"\nRunning test for n={n}, Sequence Type: {sequence_type}\n")
            start_time = time.time()
            discrepancy_history, potential_history, energy_history, red_set, blue_set = greedy_coloring(n, sequence_type)
            execution_time = time.time() - start_time

            # Execution summary
            print(f"\nExecution Summary for n={n}, Sequence Type: {sequence_type}:")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Final Discrepancy: {discrepancy_history[-1]:.6e}")
            print(f"Final Potential: {potential_history[-1]:.6e}")
            print(f"Final Energy: {energy_history[-1]:.6e}")

if __name__ == "__main__":
    main()
