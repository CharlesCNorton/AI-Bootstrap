import numpy as np
from ripser import ripser
from sklearn.preprocessing import StandardScaler
import tqdm
import scipy.stats as stats

def riemann_siegel_theta(t):
    """Enhanced Riemann-Siegel theta function with correction terms"""
    return (t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8 +
            1/(48*t) + 7/(5760*t**3) + 31/(80640*t**5))

def riemann_siegel_Z(t, terms=50):
    """Riemann-Siegel Z-function with higher precision"""
    theta_t = riemann_siegel_theta(t)
    N = int(np.sqrt(t/(2*np.pi)))

    # Main sum
    main_sum = 0
    for n in range(1, N+1):
        main_sum += np.cos(theta_t - t*np.log(n))/np.sqrt(n)

    # Correction terms
    R = np.sqrt(t/(2*np.pi))
    frac = R - N
    phi = 2*np.pi*(frac**2 - frac - 1/8)

    correction = (np.cos(phi) +
                 np.sin(phi)/(2*np.pi*R) +
                 np.cos(phi)/(4*np.pi**2*R**2))

    return 2*main_sum + (t/(2*np.pi))**(-1/4)*correction

# Known zeros of zeta function (first 8 for faster computation)
zeros = np.array([
    14.134725141734693,
    21.022039638771554,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914193
])

print("Computing derivatives for each zero...")

def compute_derivatives(t, max_order=6):
    h = 1e-7
    results = []

    for order in range(1, max_order + 1):
        # Simple central difference for derivatives
        if order == 1:
            deriv = (riemann_siegel_Z(t + h) - riemann_siegel_Z(t - h))/(2*h)
        else:
            # Higher order central differences with minimal stencil
            points = np.array([t + i*h for i in range(-order-1, order+2)])
            values = np.array([riemann_siegel_Z(p) for p in points])
            coeffs = np.zeros(len(points))
            for i in range(len(points)):
                coef = 1
                for j in range(len(points)):
                    if i != j:
                        coef *= order/(i-j)
                coeffs[i] = coef
            deriv = np.sum(coeffs * values) / h**order

        results.append(deriv)

    return np.array(results)

# Compute derivatives for all zeros
derivative_matrix = []
for zero in tqdm.tqdm(zeros, desc="Processing zeros"):
    derivs = compute_derivatives(zero)
    derivative_matrix.append(derivs)

derivative_matrix = np.array(derivative_matrix)

print("\nComputing topological features...")

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(derivative_matrix)

# Compute persistent homology
diagrams = ripser(normalized_data, maxdim=2)['dgms']

# Analysis of derivative patterns
print("\nDerivative Matrix:")
for i, zero in enumerate(zeros):
    print(f"\nZero #{i+1} (t ≈ {zero}):")
    print("Derivatives 1-6:", derivative_matrix[i])

# Analyze spacing patterns
spacings = np.diff(zeros)
normalized_spacings = spacings * np.log(zeros[1:])/(2*np.pi)

print("\nSpacing Analysis:")
print("Raw spacings:", spacings)
print("Normalized spacings:", normalized_spacings)

# Analyze derivative differences
derivative_diffs = np.diff(derivative_matrix, axis=0)
print("\nDerivative Differences:")
for order in range(derivative_matrix.shape[1]):
    print(f"\nOrder {order+1} differences:")
    print(derivative_diffs[:, order])

# Compute correlation matrix
corr_matrix = np.corrcoef(derivative_matrix.T)
print("\nDerivative Correlation Matrix:")
print(corr_matrix)

# Basic statistical measures for each derivative order
print("\nStatistical Summary by Derivative Order:")
for order in range(derivative_matrix.shape[1]):
    values = derivative_matrix[:, order]
    print(f"\nOrder {order+1}:")
    print(f"Mean: {np.mean(values)}")
    print(f"Std: {np.std(values)}")
    print(f"Min: {np.min(values)}")
    print(f"Max: {np.max(values)}")

# Analyze homological features
print("\nPersistent Homology Features:")
for dim, diagram in enumerate(diagrams):
    if len(diagram) > 0:
        print(f"\nDimension {dim}:")
        print("Birth-Death pairs:")
        print(diagram)
