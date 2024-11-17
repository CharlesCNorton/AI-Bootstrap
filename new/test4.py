import numpy as np
import mpmath as mp
from ripser import ripser

# Set high precision for calculations
mp.dps = 50  # Set the number of decimal places for precision

# Function to compute the Riemann-Siegel Z-function
def riemann_siegel_Z(t):
    return mp.zeta(0.5 + 1j * t).real

# Function to compute the nth derivative of the Riemann-Siegel Z-function
def nth_derivative_Z(t, n):
    return mp.diff(riemann_siegel_Z, t, n)

# Compute the first few non-trivial zeros of the Riemann zeta function
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

# Starting points for locating zeros (extended set)
start_points = [14, 21, 30, 40, 50, 60, 70, 85, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

# Find non-trivial zeros using refined method
zeros = find_zeros_zeta(start_points)

# Compute derivatives at these zeros (up to higher orders)
results = []
for zero in zeros:
    row = []
    row.append(float(zero))  # Add the zero itself
    for n in range(1, 9):  # Compute up to the 8th derivative
        row.append(float(nth_derivative_Z(zero, n)))
    results.append(row)

# Convert to NumPy array for enhanced persistent homology analysis
V_rho = np.array(results, dtype=float)[:, 1:]  # Exclude the 'zero' column, we just need derivatives

# Optional: Enhanced Vector Space Construction
# Normalize derivatives with respect to log(rho), but handle zero or negative values properly
# To prevent NaNs, ensure we're only taking logs of positive values
rho_values = np.array(results, dtype=float)[:, 0]
positive_indices = rho_values > 0  # Filter to keep only positive values

# Filter V_rho and rho_values to remove invalid entries
V_rho_filtered = V_rho[positive_indices, :]
rho_filtered = rho_values[positive_indices]

# Compute log(rho) for valid (positive) rho values
log_rho_filtered = np.log(rho_filtered)

# Normalize V_rho by log(rho) for enhanced analysis
V_rho_enhanced = np.array([V_rho_filtered[:, i] / log_rho_filtered for i in range(V_rho_filtered.shape[1])]).T

# Run persistent homology using ripser
result = ripser(V_rho_enhanced)

# Extract the persistence diagrams
dgms = result['dgms']

# Statistical and numerical analysis of the persistence diagrams
for i, dgm in enumerate(dgms):
    if len(dgm) == 0:
        print(f"Dimension {i}: No features detected.")
        continue

    print(f"\nDimension {i}:")
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    lifetimes = deaths - births

    # Numerical details
    print(f"Number of features: {len(dgm)}")
    print(f"Average birth time: {np.mean(births):.5f}")
    print(f"Average death time: {np.mean(deaths):.5f}")
    print(f"Average lifetime: {np.mean(lifetimes):.5f}")
    print(f"Max lifetime: {np.max(lifetimes):.5f}")

    # Persistent pairs details
    print(f"Top birth-death pairs:")
    for b, d in dgm:
        print(f"Birth: {b:.5f}, Death: {d:.5f}, Lifetime: {d - b:.5f}")
