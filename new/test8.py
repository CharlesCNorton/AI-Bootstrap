import numpy as np  # Import numpy for numerical operations
from scipy.optimize import differential_evolution
from mpmath import zeta, re
import matplotlib.pyplot as plt
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Riemann-Siegel Z-function using mpmath
def z_function(t):
    return float(re(zeta(0.5 + 1j * t)))  # Convert to float for numerical computation

# Compute the nth derivative of Z-function numerically with adaptive step size
def z_derivative(t, n, h=1e-5):
    if n == 0:
        return z_function(t)
    return (z_derivative(t + h, n - 1, h) - z_derivative(t - h, n - 1, h)) / (2 * h)

# Approximation target (example analytic function)
def target_function(s):
    return np.exp(-s**2)  # Example: Gaussian function

# Define the approximation error
def approximation_error(coeffs, t_zero, n_terms, f, region):
    error = 0
    for s in region:
        approximation = sum(
            coeffs[i] * z_derivative(t_zero, i + 1) * np.exp(1j * t_zero * s) for i in range(n_terms)
        )
        error += abs(f(s) - approximation)**2
    logging.info(f"Current error: {error}")
    return error

# Main test function using differential evolution
def run_test(t_zero, n_terms, region):
    logging.info(f"Running test with t_zero={t_zero}, n_terms={n_terms}, region={region}")

    # Define bounds for coefficients (allowing negative and positive values)
    bounds = [(-10, 10) for _ in range(n_terms)]

    # Minimize approximation error using differential evolution
    result = differential_evolution(
        approximation_error,
        bounds,
        args=(t_zero, n_terms, target_function, region),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        disp=True,
    )

    if result.success:
        optimized_coeffs = result.x
        logging.info("Optimization succeeded!")
        logging.info(f"Optimized Coefficients: {optimized_coeffs}")
        return optimized_coeffs
    else:
        logging.error("Optimization failed.")
        logging.error(f"Reason: {result.message}")
        return None

# Debugging utilities
def validate_derivatives(t_zero, max_n):
    logging.info(f"Validating derivatives for t_zero={t_zero} up to order {max_n}")
    for n in range(1, max_n + 1):
        derivative = z_derivative(t_zero, n)
        logging.info(f"Order {n} derivative at t_zero={t_zero}: {derivative}")

# Reconstruct approximation using the optimized coefficients
def approximation(t_zero, coeffs, n_terms, region):
    approx_values = []
    for s in region:
        approx = sum(
            coeffs[i] * z_derivative(t_zero, i + 1) * np.exp(1j * t_zero * s) for i in range(n_terms)
        )
        approx_values.append(np.real(approx))  # Take real part for visualization
    return approx_values

# Define test parameters
t_zero = 14.134725141  # Example: first non-trivial zero
n_terms = 5  # Number of derivatives to consider
region = np.linspace(-1, 1, 100)  # Region of approximation in the critical strip

# Run derivative validation (optional, for debugging)
validate_derivatives(t_zero, n_terms)

# Run the test
optimized_coeffs = run_test(t_zero, n_terms, region)

# Plot results if optimization succeeded
if optimized_coeffs is not None:
    target_values = [target_function(s) for s in region]
    approx_values = approximation(t_zero, optimized_coeffs, n_terms, region)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(region, target_values, label="Target Function", linestyle='--')
    plt.plot(region, approx_values, label="Approximation", linestyle='-')
    plt.title("Target Function vs. Approximation")
    plt.xlabel("s")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid()
    plt.show()
