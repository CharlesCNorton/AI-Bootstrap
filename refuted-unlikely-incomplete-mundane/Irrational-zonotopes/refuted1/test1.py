import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft
from fractions import Fraction
import math

# Full-scale 4D simulation with an updated strategy based on previous results

# Define 4D irrational and rational generators
irrational_generator1 = np.array([np.sqrt(2), np.sqrt(3), 1, 0.5])
irrational_generator2 = np.array([np.pi, np.e, 0.5, np.sqrt(5)])
rational_generator1 = np.array([1, 2, 3, 4])
rational_generator2 = np.array([2, 1, 4, 3])

# Rational approximations of the irrational generators
rational_approx1 = np.array([1.414, 1.732, 1, 0.5])
rational_approx2 = np.array([3.141, 2.718, 0.5, 2.236])  # Approximate pi, e, sqrt(5)

# Function to count lattice points in a 4D zonotope
def count_lattice_points_4D(scale_factor, gen1, gen2, gen3, gen4, sample_size=200):
    zonotope_points = np.array([
        scale_factor * (a * gen1 + b * gen2 + c * gen3 + d * gen4)
        for a in np.linspace(0, 1, sample_size)
        for b in np.linspace(0, 1, sample_size)
        for c in np.linspace(0, 1, sample_size)
        for d in np.linspace(0, 1, sample_size)
    ])

    # Count lattice points (rounded to nearest integer points)
    lattice_points = np.round(zonotope_points).astype(int)
    unique_lattice_points = np.unique(lattice_points, axis=0)

    return len(unique_lattice_points)

# Function to run tests for irrational vs rational approximations
def run_full_4D_simulation(scales, sample_size=200):
    lattice_point_counts_irrational_4D = []
    lattice_point_counts_rational_approx_4D = []

    print("Running extended 4D simulation...")
    for scale in scales:
        count_irrational = count_lattice_points_4D(scale, irrational_generator1, irrational_generator2, rational_generator1, rational_generator2, sample_size)
        count_rational_approx = count_lattice_points_4D(scale, rational_approx1, rational_approx2, rational_generator1, rational_generator2, sample_size)

        lattice_point_counts_irrational_4D.append(count_irrational)
        lattice_point_counts_rational_approx_4D.append(count_rational_approx)

        # Print lattice point counts for each scale
        print(f"Scale {scale}: Irrational = {count_irrational}, Rational Approximation = {count_rational_approx}")

    # Calculate the perturbation in 4D
    perturbation_4D = np.array(lattice_point_counts_irrational_4D) - np.array(lattice_point_counts_rational_approx_4D)

    print("\nPerturbation results:")
    for i, scale in enumerate(scales):
        print(f"Scale {scale}: Perturbation = {perturbation_4D[i]}")

    return perturbation_4D, lattice_point_counts_irrational_4D, lattice_point_counts_rational_approx_4D

# Define larger scale factors for dilation and increase sample size for more precision
scales_4D = np.arange(1, 1001, 10)  # Extended scales up to 1000 for better analysis

# Run the extended simulation with larger sample size
perturbation_4D, irr_counts, rat_counts = run_full_4D_simulation(scales_4D, sample_size=100)

# Function for Fourier analysis on perturbation
def compute_fourier_coefficients(error_sequence):
    N = len(error_sequence)
    yf = np.fft.fft(error_sequence)
    xf = np.fft.fftfreq(N, 1)
    return xf, np.abs(yf)

# Fourier analysis on the extended perturbation data
perturbation_freqs, perturbation_amplitudes = compute_fourier_coefficients(perturbation_4D)

# Print Fourier analysis results
print("\nExtended Fourier Analysis of Perturbation:")
for i, freq in enumerate(perturbation_freqs):
    if perturbation_amplitudes[i] > 1e-5:  # Only show significant frequencies
        print(f"Frequency: {freq:.4f}, Amplitude: {perturbation_amplitudes[i]:.6f}")

# Fourier spectrum visualization (extended)
plt.figure(figsize=(10, 5))
plt.stem(perturbation_freqs, perturbation_amplitudes, 'b', markerfmt=" ", basefmt="-b")
plt.title('Fourier Coefficients for Extended 4D Perturbation')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# Analysis of perturbation scaling over time
plt.figure(figsize=(10, 5))
plt.plot(scales_4D, perturbation_4D, marker='o')
plt.title('Perturbation Growth over Extended Scales (1 to 1000)')
plt.xlabel('Scale')
plt.ylabel('Perturbation (Irrational - Rational Approximation)')
plt.show()
