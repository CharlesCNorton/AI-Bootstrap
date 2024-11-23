import numpy as np
import matplotlib.pyplot as plt

def greedy_coloring(n, sequence_type='natural'):
    """
    Implements the deterministic greedy coloring algorithm.

    Parameters:
    - n: The maximum integer to consider in the sequence.
    - sequence_type: The type of sequence to use ('natural', 'odd', 'multiples_of_3', 'primes').

    Returns:
    - discrepancies: A numpy array of discrepancies at each step.
    """
    if sequence_type == 'natural':
        sequence = np.arange(2, n+1)
    elif sequence_type == 'odd':
        sequence = np.arange(3, n*2, 2)
        sequence = sequence[sequence <= n]
    elif sequence_type == 'multiples_of_3':
        sequence = np.arange(3, n+1, 3)
    elif sequence_type == 'primes':
        sequence = sieve_of_eratosthenes(n)
    else:
        raise ValueError("Invalid sequence type. Choose from 'natural', 'odd', 'multiples_of_3', 'primes'.")

    S_red = 0.0   # Sum of reciprocals in the red set
    S_blue = 0.0  # Sum of reciprocals in the blue set
    D_n = 0.0     # Discrepancy
    discrepancies = []

    for i in sequence:
        recip = 1.0 / i
        # Compute potential discrepancies for both choices
        D_red = abs((S_red + recip) - S_blue)
        D_blue = abs(S_red - (S_blue + recip))
        # Assign to the color that minimizes the discrepancy
        if D_red <= D_blue:
            S_red += recip
        else:
            S_blue += recip
        D_n = S_red - S_blue
        discrepancies.append(D_n)

    discrepancies = np.array(discrepancies)
    return discrepancies

def sieve_of_eratosthenes(max_num):
    """
    Generates all prime numbers up to max_num using the Sieve of Eratosthenes.

    Parameters:
    - max_num: The maximum number up to which to generate primes.

    Returns:
    - primes: A numpy array of prime numbers up to max_num.
    """
    is_prime = np.ones(max_num + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(max_num)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]
    return primes

def compute_statistics(discrepancies):
    """
    Computes statistical measures of the discrepancies.

    Parameters:
    - discrepancies: A numpy array of discrepancies at each step.

    Returns:
    - stats: A dictionary containing mean, variance, max, and min of discrepancies.
    """
    D_n = discrepancies
    stats = {
        'mean_discrepancy': np.mean(D_n),
        'variance': np.var(D_n),
        'max_discrepancy': np.max(D_n),
        'min_discrepancy': np.min(D_n),
        'final_discrepancy': D_n[-1]
    }
    return stats

def plot_discrepancies(discrepancies, sequence_type):
    """
    Plots the evolution of the discrepancy over time.

    Parameters:
    - discrepancies: A numpy array of discrepancies at each step.
    - sequence_type: The type of sequence used.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(discrepancies, label='Discrepancy D_n')
    plt.hlines([1, -1], 0, len(discrepancies), colors='red', linestyles='dashed', label='Bounds C=1')
    plt.title(f'Discrepancy Evolution for {sequence_type.capitalize()} Numbers Sequence')
    plt.xlabel('Steps')
    plt.ylabel('Discrepancy D_n')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    n = 1000000000  # Adjust n as needed
    sequence_type = 'primes'  # Choose from 'natural', 'odd', 'multiples_of_3', 'primes'

    discrepancies = greedy_coloring(n, sequence_type)
    stats = compute_statistics(discrepancies)

    # Output the statistics
    print(f"Statistics for {sequence_type.capitalize()} Numbers Sequence with n = {n}:")
    print(f"Mean Discrepancy: {stats['mean_discrepancy']}")
    print(f"Variance: {stats['variance']}")
    print(f"Maximum Discrepancy: {stats['max_discrepancy']}")
    print(f"Minimum Discrepancy: {stats['min_discrepancy']}")
    print(f"Final Discrepancy D_n: {stats['final_discrepancy']}")

    # Optionally, plot the discrepancies
    plot_discrepancies(discrepancies, sequence_type)

if __name__ == "__main__":
    main()
