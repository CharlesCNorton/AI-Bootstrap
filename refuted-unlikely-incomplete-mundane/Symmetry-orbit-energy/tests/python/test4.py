import numpy as np
from scipy.integrate import nquad

# Step 1: Define the density functions for root clustering
def f_e6(x):
    """Density function for E6."""
    norm_x = np.linalg.norm(x)
    return np.exp(-0.1 * norm_x) if norm_x > 0 else 0.0

def f_e8(x):
    """Density function for E8."""
    norm_x = np.linalg.norm(x)
    return np.exp(-0.15 * norm_x) if norm_x > 0 else 0.0

# Step 2: Define the entropy integrand function
def entropy_integrand_e6(*x):
    f_val = f_e6(x)
    return -f_val * np.log(f_val) if f_val > 0 else 0.0

def entropy_integrand_e8(*x):
    f_val = f_e8(x)
    return -f_val * np.log(f_val) if f_val > 0 else 0.0

# Step 3: Set the integration bounds for E6 and E8
bounds_e6 = [(-5, 5)] * 6  # 6-dimensional bounds for E6
bounds_e8 = [(-5, 5)] * 8  # 8-dimensional bounds for E8

# Step 4: Perform the numerical integration using SciPy's nquad for E6
print("Calculating Symmetry Orbit Entropy for E6 using SciPy...")

result_e6, error_e6 = nquad(entropy_integrand_e6, bounds_e6)

print(f"Symmetry Orbit Entropy for E6: {result_e6:.5e} with error estimate {error_e6:.5e}")

# Step 5: Perform the numerical integration using SciPy's nquad for E8
print("Calculating Symmetry Orbit Entropy for E8 using SciPy...")

result_e8, error_e8 = nquad(entropy_integrand_e8, bounds_e8)

print(f"Symmetry Orbit Entropy for E8: {result_e8:.5e} with error estimate {error_e8:.5e}")

# Step 6: Compare the SOE values to determine isomorphism potential
threshold = 0.1
soe_difference = abs(result_e6 - result_e8)

if soe_difference < threshold:
    print("E6 and E8 may be isomorphic based on the SOE measure.")
else:
    print("E6 and E8 are not isomorphic based on the SOE measure.")
