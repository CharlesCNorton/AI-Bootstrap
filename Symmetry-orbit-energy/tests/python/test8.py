import numpy as np
import scipy
from scipy.stats import qmc
import math

# ---------- STEP 1: Define Adaptive Monte Carlo Functions for SOE Calculations ----------
# These functions calculate Symmetry Orbit Entropy (SOE) using adaptive Monte Carlo for different group types.

# 1. SU(5) - Unitary Group (Entropy Calculation)
def adaptive_monte_carlo_su5(num_samples=100000):
    sampler = qmc.Sobol(d=5, scramble=True)
    samples = sampler.random_base2(m=int(np.log2(num_samples)))

    # Scale samples to appropriate ranges [0, π] for each dimension
    scaled_samples = np.pi * samples

    # Define the density for entropy calculation
    def density_su5(phi1, phi2, phi3, phi4, phi5):
        return np.sin(phi1) * np.sin(phi2) * np.sin(phi3) * np.sin(phi4) * np.sin(phi5)

    # Calculate entropy values for each sample
    entropy_values = []
    for sample in scaled_samples:
        phi1, phi2, phi3, phi4, phi5 = sample
        density_value = density_su5(phi1, phi2, phi3, phi4, phi5)
        if density_value > 0:
            entropy_values.append(-density_value * np.log(density_value))
        else:
            entropy_values.append(0)

    # Estimate entropy as the average value times the volume
    entropy_estimate = np.mean(entropy_values) * (np.pi ** 5)
    return entropy_estimate

# 2. SO(10) - Orthogonal Group (Entropy Calculation)
def adaptive_monte_carlo_so10(num_samples=100000):
    sampler = qmc.Sobol(d=10, scramble=True)
    samples = sampler.random_base2(m=int(np.log2(num_samples)))

    # Scale samples to the appropriate range [0, π] for each dimension
    scaled_samples = np.pi * samples

    # Define the density for entropy calculation
    def density_so10(phi):
        return np.prod(np.sin(phi))

    # Calculate entropy values for each sample
    entropy_values = []
    for sample in scaled_samples:
        density_value = density_so10(sample)
        if density_value > 0:
            entropy_values.append(-density_value * np.log(density_value))
        else:
            entropy_values.append(0)

    # Estimate entropy as the average value times the volume
    entropy_estimate = np.mean(entropy_values) * (np.pi ** 10)
    return entropy_estimate

# 3. E6 - Exceptional Group (Entropy Calculation)
def adaptive_monte_carlo_e6(num_samples=100000):
    sampler = qmc.Sobol(d=6, scramble=True)
    samples = sampler.random_base2(m=int(np.log2(num_samples)))

    # Scale samples to appropriate ranges [0, π] for each dimension
    scaled_samples = np.pi * samples

    # Define the density for entropy calculation
    def density_e6(phi1, phi2, phi3, phi4, phi5, phi6):
        return np.sin(phi1) * np.sin(phi2) * np.sin(phi3) * np.sin(phi4) * np.sin(phi5) * np.sin(phi6)

    # Calculate entropy values for each sample
    entropy_values = []
    for sample in scaled_samples:
        phi1, phi2, phi3, phi4, phi5, phi6 = sample
        density_value = density_e6(phi1, phi2, phi3, phi4, phi5, phi6)
        if density_value > 0:
            entropy_values.append(-density_value * np.log(density_value))
        else:
            entropy_values.append(0)

    # Estimate entropy as the average value times the volume
    entropy_estimate = np.mean(entropy_values) * (np.pi ** 6)
    return entropy_estimate

# 4. Sp(4) - Symplectic Group (Entropy Calculation)
def adaptive_monte_carlo_sp4(num_samples=100000):
    sampler = qmc.Sobol(d=4, scramble=True)
    samples = sampler.random_base2(m=int(np.log2(num_samples)))

    # Scale samples to appropriate ranges [0, π] for each dimension
    scaled_samples = np.pi * samples

    # Define the density for entropy calculation
    def density_sp4(phi1, phi2, phi3, phi4):
        return np.sin(phi1) * np.sin(phi2) * np.sin(phi3) * np.sin(phi4)

    # Calculate entropy values for each sample
    entropy_values = []
    for sample in scaled_samples:
        phi1, phi2, phi3, phi4 = sample
        density_value = density_sp4(phi1, phi2, phi3, phi4)
        if density_value > 0:
            entropy_values.append(-density_value * np.log(density_value))
        else:
            entropy_values.append(0)

    # Estimate entropy as the average value times the volume
    entropy_estimate = np.mean(entropy_values) * (np.pi ** 4)
    return entropy_estimate

# 5. SO(3,1) - Non-Compact Group (Entropy Calculation)
def adaptive_monte_carlo_so31(num_samples=100000):
    sampler = qmc.Sobol(d=4, scramble=True)
    samples = sampler.random_base2(m=int(np.log2(num_samples)))

    # Scale samples to appropriate ranges [0, 10] for theta1, theta2, theta3 and [0, 2π] for alpha
    scaled_samples = samples.copy()
    scaled_samples[:, 0:3] *= 10  # Scale the first three dimensions to [0, 10]
    scaled_samples[:, 3] *= 2 * np.pi  # Scale the last dimension to [0, 2π]

    # Define the density for entropy calculation
    def density_so31(theta1, theta2, theta3, alpha):
        return np.sinh(theta1) * np.sinh(theta2) * np.sinh(theta3)

    # Calculate entropy values for each sample
    entropy_values = []
    for sample in scaled_samples:
        theta1, theta2, theta3, alpha = sample
        density_value = density_so31(theta1, theta2, theta3, alpha)
        if density_value > 0:
            entropy_values.append(-density_value * np.log(density_value))
        else:
            entropy_values.append(0)

    # Estimate entropy as the average value times the volume
    entropy_estimate = np.mean(entropy_values) * (10 * 10 * 10 * 2 * np.pi)
    return entropy_estimate


# ---------- STEP 2: Temperature-Dependent Entropy Analysis ----------
# This function calculates the temperature-dependent entropy for each group function.

def temperature_dependent_entropy(group_function, temperature_scales, num_samples=100000):
    entropy_results = []

    for T in temperature_scales:
        # Perform Monte Carlo integration for entropy calculation at each temperature scale
        entropy_value = group_function(num_samples)
        # Apply a temperature scaling factor (e.g., entropy scales with inverse temperature)
        scaled_entropy = entropy_value / T
        entropy_results.append(scaled_entropy)

    return entropy_results

# Define temperature scales to simulate cooling from high to low temperature
temperature_scales = np.linspace(10, 1, 10)  # Cooling from T = 10 to T = 1 in 10 steps

# ---------- STEP 3: Perform All Entropy Calculations for Candidate Groups ----------
# Candidate groups: SU(5), SO(10), E6, Sp(4), SO(3,1)

# Calculate entropy for each group using Monte Carlo methods
entropy_su5 = adaptive_monte_carlo_su5()
entropy_so10 = adaptive_monte_carlo_so10()
entropy_e6 = adaptive_monte_carlo_e6()
entropy_sp4 = adaptive_monte_carlo_sp4()
entropy_so31 = adaptive_monte_carlo_so31()

# Calculate temperature-dependent entropy for each candidate symmetry group
entropy_temperature_su5 = temperature_dependent_entropy(adaptive_monte_carlo_su5, temperature_scales)
entropy_temperature_so10 = temperature_dependent_entropy(adaptive_monte_carlo_so10, temperature_scales)
entropy_temperature_e6 = temperature_dependent_entropy(adaptive_monte_carlo_e6, temperature_scales)
entropy_temperature_sp4 = temperature_dependent_entropy(adaptive_monte_carlo_sp4, temperature_scales)
entropy_temperature_so31 = temperature_dependent_entropy(adaptive_monte_carlo_so31, temperature_scales)

# Compile results into a comprehensive dictionary
results = {
    "Initial Entropy Calculations": {
        "SU(5)": entropy_su5,
        "SO(10)": entropy_so10,
        "E6": entropy_e6,
        "Sp(4)": entropy_sp4,
        "SO(3,1)": entropy_so31,
    },
    "Temperature-Dependent Entropy": {
        "SU(5)": entropy_temperature_su5,
        "SO(10)": entropy_temperature_so10,
        "E6": entropy_temperature_e6,
        "Sp(4)": entropy_temperature_sp4,
        "SO(3,1)": entropy_temperature_so31,
    }
}

# ---------- STEP 4: Output Results ----------
# Output all the results, including initial entropy and temperature-dependent analysis.
for group, entropy in results["Initial Entropy Calculations"].items():
    print(f"{group} Initial Entropy: {entropy:.5f}")

print("\nTemperature-Dependent Entropy Analysis:\n")
for group, entropies in results["Temperature-Dependent Entropy"].items():
    print(f"{group}:")
    for i, T in enumerate(temperature_scales):
        print(f"  Temperature T = {T:.1f}, Entropy = {entropies[i]:.5f}")
    print()
