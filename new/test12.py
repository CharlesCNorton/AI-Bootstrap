import re
import numpy as np
from scipy.stats import ttest_1samp
from mpmath import mp, gamma as mp_gamma, arg as mp_arg, power as mp_power

# Set higher precision for numerical accuracy
mp.dps = 150  # Precision to 150 decimal places

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
def compute_derivative(f, t, n, h=1e-6):
    """
    Compute the n-th numerical derivative of a function f at t using finite differences.
    """
    t = mp.mpf(t)  # High precision for t
    coeffs = np.array([(-1) ** (n + k) * np.math.comb(n, k) for k in range(n + 1)])
    return np.sum(coeffs * np.array([f(t + k * h - n * h / 2) for k in range(n + 1)])) / h**n

# Quasi-periodicity analysis
def analyze_quasi_periodicity(gamma_values, order=1):
    """Analyze phases for quasi-periodicity in derivatives."""
    phases = []
    for gamma in gamma_values:
        dZ_val = compute_derivative(Z, gamma, order)
        phases.append(np.angle(dZ_val))  # Extract phase information
    return phases

# Arithmetic structure analysis
def analyze_arithmetic_structure(gamma_values, order=1):
    """Analyze modular relationships in derivative ratios."""
    ratios = []
    for gamma in gamma_values:
        dZ1 = compute_derivative(Z, gamma, order)
        dZ2 = compute_derivative(Z, gamma, order + 1)
        ratios.append(abs(dZ2) / abs(dZ1))  # Ratio of successive derivatives
    return ratios

# Main function
def main():
    file_path = r"C:\Users\cnort\Desktop\zeros4.txt"  # Path to your zeros file
    gamma_values = parse_zeros(file_path)

    # Expand gamma range (e.g., 31-50)
    expanded_gamma_values = gamma_values[30:50]

    # Scaling Test for First Derivative
    print("Scaling Test Results for Expanded Gamma Range:")
    scaling_results = []
    for order in range(1, 5):  # First to Fourth Derivative
        scaling_results.append(analyze_arithmetic_structure(expanded_gamma_values, order=order))
        print(f"Order-{order}: {scaling_results}")

    # Quasi-Periodicity Analysis
    print("\nQuasi-Periodicity Analysis:")
    quasi_results = analyze_quasi_periodicity(expanded_gamma_values, order=1)
    print(quasi_results)

if __name__ == "__main__":
    main()
