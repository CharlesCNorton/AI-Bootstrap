import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_path_space(x, y, dim):
    """Create path space matrix as defined in the paper"""
    dist = np.linalg.norm(x - y)
    epsilon = 0.01 / (1 + 0.01 * dim)
    perturbation = np.random.uniform(-1, 1, (dim, dim))
    return np.eye(dim) + epsilon * np.exp(-0.3 * dist) * perturbation

def measure_properties(dim, samples=1000):
    """Measure reflexivity, symmetry, and transitivity with proper normalization"""
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
    """Exponential decay function: a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def comprehensive_analysis(trials=10):
    dimensions = range(1, 31)
    all_results = {
        'R': np.zeros((trials, len(dimensions))),
        'S': np.zeros((trials, len(dimensions))),
        'T': np.zeros((trials, len(dimensions))),
        'R_std': np.zeros((trials, len(dimensions))),
        'S_std': np.zeros((trials, len(dimensions))),
        'T_std': np.zeros((trials, len(dimensions)))
    }

    print("Running trials...")
    for trial in tqdm(range(trials)):
        for i, d in enumerate(dimensions):
            R, R_std, S, S_std, T, T_std = measure_properties(d)
            all_results['R'][trial, i] = R
            all_results['S'][trial, i] = S
            all_results['T'][trial, i] = T
            all_results['R_std'][trial, i] = R_std
            all_results['S_std'][trial, i] = S_std
            all_results['T_std'][trial, i] = T_std

    # Calculate means and confidence intervals
    results = {}
    for prop in ['R', 'S', 'T']:
        results[prop] = {
            'mean': np.mean(all_results[prop], axis=0),
            'std': np.std(all_results[prop], axis=0),
            'ci': stats.t.interval(0.95, trials-1,
                                 loc=np.mean(all_results[prop], axis=0),
                                 scale=stats.sem(all_results[prop], axis=0))
        }

    # Fit decay curves
    decay_params = {}
    for prop in ['R', 'S', 'T']:
        popt, _ = curve_fit(exp_decay, dimensions, results[prop]['mean'],
                           p0=[0.02, 0.1, 0.95])
        decay_params[prop] = popt

    # Calculate ratio evolution
    S_R_ratios = results['S']['mean'] / results['R']['mean']
    ratio_fit = np.polyfit(dimensions, S_R_ratios, 1)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Property values
    plt.subplot(2, 1, 1)
    colors = {'R': 'b', 'S': 'r', 'T': 'g'}
    for prop in ['R', 'S', 'T']:
        plt.plot(dimensions, results[prop]['mean'], color=colors[prop], linestyle='-', label=prop)
        plt.fill_between(dimensions,
                        results[prop]['ci'][0],
                        results[prop]['ci'][1],
                        color=colors[prop], alpha=0.2)

        # Plot fitted decay curves
        a, b, c = decay_params[prop]
        plt.plot(dimensions, exp_decay(np.array(dimensions), a, b, c),
                color=colors[prop], linestyle='--', alpha=0.5)

    plt.xlabel('Dimension')
    plt.ylabel('Property Value')
    plt.title('Path Space Properties vs Dimension')
    plt.legend()
    plt.grid(True)

    # Ratio evolution
    plt.subplot(2, 1, 2)
    plt.plot(dimensions, S_R_ratios, color='k', linestyle='-', label='S/R Ratio')
    plt.plot(dimensions, np.polyval(ratio_fit, dimensions), color='r', linestyle='--',
             label=f'Linear fit (slope={ratio_fit[0]:.6f})')
    plt.xlabel('Dimension')
    plt.ylabel('S/R Ratio')
    plt.title('Property Ratio Evolution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Statistical analysis
    hierarchy_violations = np.sum(~((results['S']['mean'] > results['T']['mean']) &
                                  (results['T']['mean'] > results['R']['mean'])))

    # Asymptotic values (last 5 dimensions)
    asymp_window = 5
    asymp_values = {
        prop: {
            'mean': np.mean(results[prop]['mean'][-asymp_window:]),
            'std': np.std(results[prop]['mean'][-asymp_window:]),
            'ci': stats.t.interval(0.95, asymp_window-1,
                                 loc=np.mean(results[prop]['mean'][-asymp_window:]),
                                 scale=stats.sem(results[prop]['mean'][-asymp_window:]))
        }
        for prop in ['R', 'S', 'T']
    }

    return {
        'decay_params': decay_params,
        'ratio_slope': ratio_fit[0],
        'hierarchy_violations': hierarchy_violations,
        'asymptotic_values': asymp_values,
        'raw_results': results
    }

if __name__ == "__main__":
    np.random.seed(42)
    results = comprehensive_analysis()

    print("\nDecay Parameters:")
    for prop in ['R', 'S', 'T']:
        a, b, c = results['decay_params'][prop]
        print(f"{prop}: {a:.6f} * exp(-{b:.6f} * d) + {c:.6f}")

    print(f"\nRatio Evolution Slope: {results['ratio_slope']:.6f}")
    print(f"Hierarchy Violations: {results['hierarchy_violations']}")

    print("\nAsymptotic Values:")
    for prop in ['R', 'S', 'T']:
        values = results['asymptotic_values'][prop]
        print(f"{prop}_∞ = {values['mean']:.6f} ± {values['std']:.6f}")
        print(f"    95% CI: [{values['ci'][0]:.6f}, {values['ci'][1]:.6f}]")
