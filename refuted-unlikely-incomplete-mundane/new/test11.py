import re
import numpy as np
from mpmath import mp, gamma as mp_gamma, arg as mp_arg, power as mp_power

# Set higher precision for numerical accuracy
mp.dps = 100  # Precision to 100 decimal places

# Parsing gamma values from the zeros4.txt file
def parse_zeros(file_path):
    """Extract gamma values from the zeros4.txt file."""
    gamma_values = []
    with open(file_path, 'r') as file:
        for line in file:
            # Match numerical lines (excluding headers and descriptions)
            match = re.match(r"^\s*([\d.]+)\s*$", line)
            if match:
                gamma_values.append(float(match.group(1)))
    return gamma_values

# Define the Riemann-Siegel theta function
def theta(t):
    """Riemann-Siegel theta function."""
    t = mp.mpf(t)  # Ensure high-precision input
    return mp_arg(mp_gamma(0.25 + 0.5j * t)) - 0.5 * t * mp.log(mp.pi)

# Sieve of Eratosthenes to generate primes
def sieve_of_eratosthenes(limit):
    """Generate all prime numbers up to the specified limit using the Sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not primes
    for start in range(2, int(limit**0.5) + 1):
        if sieve[start]:
            for multiple in range(start * start, limit + 1, start):
                sieve[multiple] = False
    return [num for num, is_prime in enumerate(sieve) if is_prime]

# Zeta approximation using sieve
def zeta_sieve_approx(s, limit=10**6):
    """
    Approximation of the Riemann zeta function using a sieve to sum over primes.
    """
    primes = sieve_of_eratosthenes(limit)
    s = mp.mpc(s)  # Ensure high precision
    return sum(mp_power(p, -s) for p in primes)

# Define the Riemann-Siegel Z function
def Z(t, sieve_limit=10**6):
    """Riemann-Siegel Z function using sieve approximation for zeta."""
    t = mp.mpf(t)  # Ensure high-precision input
    zeta_value = zeta_sieve_approx(0.5 + 1j * t, limit=sieve_limit)
    return np.real(complex(mp.exp(1j * theta(t)) * zeta_value))

# Numerical derivatives of Z(t)
def compute_derivative(f, t, n, h=1e-5):
    """
    Compute the n-th numerical derivative of a function f at t using finite differences.
    """
    t = mp.mpf(t)  # High precision for t
    coeffs = np.array([(-1) ** (n + k) * np.math.comb(n, k) for k in range(n + 1)])
    return np.sum(coeffs * np.array([f(t + k * h - n * h / 2) for k in range(n + 1)])) / h**n

# Scaling test
def scaling_test(gamma_values):
    """Verify scaling for Z'(gamma)."""
    results = []
    for gamma in gamma_values:  # Analyze the full range of gamma values
        Z_val = Z(gamma)
        dZ_val = compute_derivative(Z, gamma, 1)
        scaling = abs(dZ_val) / gamma**0.5  # Scaling prediction
        results.append((gamma, abs(Z_val), abs(dZ_val), scaling))
    return results

# Main function
def main():
    file_path = r"C:\Users\cnort\Desktop\zeros4.txt"  # Path to your zeros file
    gamma_values = parse_zeros(file_path)

    # Select a larger range of gamma values (e.g., 11-20 instead of just the first 10)
    extended_gamma_values = gamma_values[10:20]

    # Scaling Test with Extended Gamma Values
    scaling_results = scaling_test(extended_gamma_values)
    print("Scaling Test Results for Larger Gamma Values:")
    print("Gamma\t\t\t|Z(gamma)|\t\t|Z'(gamma)|\t\tScaling")
    for gamma, Z_val, dZ_val, scaling in scaling_results:
        print(f"{gamma:.6f}\t\t{Z_val:.6f}\t\t{dZ_val:.6f}\t\t{scaling:.6f}")

if __name__ == "__main__":
    main()
