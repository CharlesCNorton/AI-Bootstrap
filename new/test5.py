import numpy as np
import mpmath as mp
import csv

# Set very low precision to manage memory
mp.dps = 10  # Set the number of decimal places for precision to a very low value

# Function to compute the Riemann-Siegel Z-function
def riemann_siegel_Z(t):
    return mp.zeta(0.5 + 1j * t).real

# Function to compute the nth derivative of the Riemann-Siegel Z-function
def nth_derivative_Z(t, n):
    return mp.diff(riemann_siegel_Z, t, n)

# Compute a limited number of non-trivial zeros of the Riemann zeta function
def find_zeros_zeta(start_points, max_attempts=5):
    zeros = []
    for t in start_points:
        attempt = 0
        found = False
        while attempt < max_attempts and not found:
            try:
                zero = mp.findroot(lambda s: mp.zeta(s), 0.5 + 1j * t, solver='newton')
                zeros.append(zero.imag)  # Record only the imaginary part of the zero
                found = True
            except ValueError:
                t += 1  # Adjust starting point slightly and retry
                attempt += 1
    return zeros

# Reduce the number of zeros computed to manage memory
start_points = list(range(14, 64, 5))

# Find non-trivial zeros
zeros = find_zeros_zeta(start_points)

# Compute derivatives for these zeros and save to a CSV file
with open('zeta_derivatives.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["zero", "dZ/dt", "d^2Z/dt^2", "d^3Z/dt^3", "d^4Z/dt^4"])  # Header
    for zero in zeros:
        row = [float(zero)]
        for n in range(1, 5):  # Compute up to the 4th derivative
            row.append(float(nth_derivative_Z(zero, n)))
        writer.writerow(row)
