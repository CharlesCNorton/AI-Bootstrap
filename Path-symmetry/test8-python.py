import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_path_space(x, y, dim):
    """Create path space matrix with controlled perturbation"""
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * dim)
    perturbation = np.random.uniform(-1, 1, (dim, dim))
    return np.eye(dim) + epsilon * np.exp(-0.3 * dist) * perturbation

def measure_properties(dim, samples=1000):
    """Measure reflexivity, symmetry, transitivity with proper normalization"""
    R_values = []
    S_values = []
    T_values = []

    for _ in range(samples):
        x = np.random.uniform(-1, 1, dim)
        y = np.random.uniform(-1, 1, dim)
        z = np.random.uniform(-1, 1, dim)

        P_xx = create_path_space(x, x, dim)
        R = 1 - (np.linalg.norm(P_xx - np.eye(dim), 'fro') /
                 np.linalg.norm(np.eye(dim), 'fro'))

        P_xy = create_path_space(x, y, dim)
        P_yx = create_path_space(y, x, dim)
        S = 1 - (np.linalg.norm(P_xy - P_yx.T, 'fro') /
                 (np.linalg.norm(P_xy, 'fro') + np.linalg.norm(P_yx, 'fro')))

        P_yz = create_path_space(y, z, dim)
        P_xz = create_path_space(x, z, dim)
        composition = P_xy @ P_yz
        T = 1 - (np.linalg.norm(composition - P_xz, 'fro') /
                 (np.linalg.norm(composition, 'fro') + np.linalg.norm(P_xz, 'fro')))

        R_values.append(R)
        S_values.append(S)
        T_values.append(T)

    return np.mean(R_values), np.std(R_values), np.mean(S_values), np.std(S_values), np.mean(T_values), np.std(T_values)

def exp_decay(x, a, b, c):
    """Exponential decay function with parameters"""
    return a * np.exp(-b * x) + c

def coherence_bound(dimension, tolerance, decay_rate):
    """Calculate bounded coherence conditions needed"""
    return int(np.ceil(-np.log(tolerance) / decay_rate))

def test_decay_rates(base_rates, variation=0.1, dims=range(1, 31)):
    """Test decay rates with variations"""
    R_base, S_base, T_base = base_rates
    variations = np.arange(1 - variation, 1 + variation + 0.1, 0.1)

    results = {}
    for var in variations:
        R_rate = R_base * var
        S_rate = S_base * var
        T_rate = T_base * var

        # Calculate coherence bounds
        tolerance = 0.01
        R_bound = coherence_bound(max(dims), tolerance, R_rate)
        S_bound = coherence_bound(max(dims), tolerance, S_rate)
        T_bound = coherence_bound(max(dims), tolerance, T_rate)

        results[f'variation_{var:.1f}'] = {
            'rates': (R_rate, S_rate, T_rate),
            'bounds': (R_bound, S_bound, T_bound)
        }

    return results

def analyze_coherence():
    """Main analysis function"""
    # Base decay rates from our measurements
    base_rates = (0.086160, 0.765047, 0.766237)

    # Test variations
    rate_results = test_decay_rates(base_rates)

    # Measure actual properties
    print("Measuring properties...")
    dimensions = range(1, 31)
    measured_results = {
        'R': [], 'S': [], 'T': [],
        'R_std': [], 'S_std': [], 'T_std': []
    }

    for d in tqdm(dimensions):
        R, R_std, S, S_std, T, T_std = measure_properties(d)
        measured_results['R'].append(R)
        measured_results['S'].append(S)
        measured_results['T'].append(T)
        measured_results['R_std'].append(R_std)
        measured_results['S_std'].append(S_std)
        measured_results['T_std'].append(T_std)

    # Fit decay curves to measured data
    popt_R, _ = curve_fit(exp_decay, dimensions, measured_results['R'])
    popt_S, _ = curve_fit(exp_decay, dimensions, measured_results['S'])
    popt_T, _ = curve_fit(exp_decay, dimensions, measured_results['T'])

    # Plot results
    plt.figure(figsize=(15, 10))

    # Property measurements and fits
    plt.subplot(2, 1, 1)
    colors = {'R': 'b', 'S': 'r', 'T': 'g'}
    labels = {'R': 'Reflexivity', 'S': 'Symmetry', 'T': 'Transitivity'}

    for prop in ['R', 'S', 'T']:
        plt.errorbar(list(dimensions), measured_results[prop],
                    yerr=measured_results[f'{prop}_std'],
                    color=colors[prop], fmt='o', label=f'{labels[prop]} (measured)')

    # Plot fitted curves
    x_fit = np.linspace(min(dimensions), max(dimensions), 100)
    plt.plot(x_fit, exp_decay(x_fit, *popt_R), 'b--', label='R fit')
    plt.plot(x_fit, exp_decay(x_fit, *popt_S), 'r--', label='S fit')
    plt.plot(x_fit, exp_decay(x_fit, *popt_T), 'g--', label='T fit')

    plt.xlabel('Dimension')
    plt.ylabel('Property Value')
    plt.title('Path Space Properties vs Dimension')
    plt.legend()
    plt.grid(True)

    # Coherence bounds for different variations
    plt.subplot(2, 1, 2)
    variations = sorted(rate_results.keys())
    var_values = [float(v.split('_')[1]) for v in variations]

    # Ensure all arrays have the same length
    bounds_R = []
    bounds_S = []
    bounds_T = []

    for v in variations:
        bounds = rate_results[v]['bounds']
        bounds_R.append(bounds[0])
        bounds_S.append(bounds[1])
        bounds_T.append(bounds[2])

    plt.plot(var_values, bounds_R, 'b-o', label=f'{labels["R"]} coherence bound')
    plt.plot(var_values, bounds_S, 'r-o', label=f'{labels["S"]} coherence bound')
    plt.plot(var_values, bounds_T, 'g-o', label=f'{labels["T"]} coherence bound')

    plt.xlabel('Rate Variation Factor')
    plt.ylabel('Required Coherence Conditions')
    plt.title('Coherence Bounds vs Rate Variation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print numerical results
    print("\nFitted Decay Parameters:")
    for prop, popt in zip(['R', 'S', 'T'], [popt_R, popt_S, popt_T]):
        print(f"{prop}: {popt[0]:.6f} * exp(-{popt[1]:.6f} * d) + {popt[2]:.6f}")

    print("\nCoherence Bounds for Different Rate Variations:")
    for var in variations:
        bounds = rate_results[var]['bounds']
        print(f"\nVariation factor {float(var.split('_')[1]):.1f}:")
        print(f"R bound: {bounds[0]}")
        print(f"S bound: {bounds[1]}")
        print(f"T bound: {bounds[2]}")

if __name__ == "__main__":
    np.random.seed(42)
    analyze_coherence()
