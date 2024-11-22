import numpy as np
from scipy.optimize import differential_evolution
from mpmath import zeta, re
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

# Define the approximation error with regularization
def approximation_error(coeffs, t_zero, n_terms, f, region, reg_lambda=0.01):
    error = 0
    for s in region:
        approximation = sum(
            coeffs[i] * z_derivative(t_zero, i + 1) * np.exp(1j * t_zero * s) for i in range(n_terms)
        )
        error += abs(f(s) - approximation)**2
    # Add regularization penalty to control overfitting
    penalty = reg_lambda * np.sum(np.abs(coeffs))
    error += penalty
    return error

# Reconstruct approximation using the optimized coefficients
def approximation(t_zero, coeffs, n_terms, region):
    approx_values = []
    for s in region:
        approx = sum(
            coeffs[i] * z_derivative(t_zero, i + 1) * np.exp(1j * t_zero * s) for i in range(n_terms)
        )
        approx_values.append(np.real(approx))  # Take real part for visualization
    return approx_values

# Main test function using differential evolution
def run_test(t_zero, n_terms, region, reg_lambda):
    logging.info(f"Running test with t_zero={t_zero}, n_terms={n_terms}, region={region}")

    # Define bounds for coefficients (allowing negative and positive values)
    bounds = [(-10, 10) for _ in range(n_terms)]

    # Minimize approximation error using differential evolution
    result = differential_evolution(
        approximation_error,
        bounds,
        args=(t_zero, n_terms, target_function, region, reg_lambda),
        strategy='best1bin',
        maxiter=2000,
        popsize=20,
        tol=1e-8,
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

# Summary statistics calculation
def calculate_summary_statistics(target_values, approx_values):
    differences = np.array(target_values) - np.array(approx_values)
    mse = np.mean(differences**2)  # Mean Squared Error
    mae = np.mean(np.abs(differences))  # Mean Absolute Error
    max_error = np.max(np.abs(differences))  # Maximum Error

    print("\n--- Summary Statistics ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Maximum Absolute Error: {max_error:.6f}")

# Define test parameters
t_zero = 14.134725141  # Example: first non-trivial zero
n_terms = 10  # Increased number of derivatives to consider
region = np.linspace(-0.5, 0.5, 100)  # Narrowed region for better focus
reg_lambda = 0.1  # Regularization strength to prevent large coefficients

# Run the test
optimized_coeffs = run_test(t_zero, n_terms, region, reg_lambda)

# Generate summary statistics if optimization succeeded
if optimized_coeffs is not None:
    target_values = [target_function(s) for s in region]
    approx_values = approximation(t_zero, optimized_coeffs, n_terms, region)
    calculate_summary_statistics(target_values, approx_values)
