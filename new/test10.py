import numpy as np
from mpmath import mp, gamma as mp_gamma, arg as mp_arg, power as mp_power

# Set arbitrary precision
mp.dps = 50  # Set precision to 50 decimal places

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

# Example gamma values from zeros4.txt
gamma_values = [
    mp.mpf('144176897509546973538.49806962'),
    mp.mpf('144176897509546973538.69355122'),
    mp.mpf('144176897509546973538.75465657'),
    mp.mpf('144176897509546973538.85404881'),
]

# Compute Z(t) and its first derivative for gamma values
results = []
for gamma in gamma_values:
    Z_val = Z(gamma)
    dZ_val = compute_derivative(Z, gamma, 1)
    results.append((gamma, Z_val, dZ_val))

# Display results
print("Gamma Values and Computed Results:")
print("Gamma\t\t\t\tZ(gamma)\t\t\t\tZ'(gamma)")
for gamma, Z_val, dZ_val in results:
    print(f"{gamma}\t\t{Z_val:.6f}\t\t{dZ_val:.6f}")
