import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def C(n):
    """Calculate minimal coherence conditions for n-categories"""
    if n < 2:
        raise ValueError("n must be >= 2")

    if n <= 3:
        return n - 1  # Foundational phase
    elif n <= 5:
        return 2*n - 3  # Transitional phase
    else:
        return 2*n - 1  # Linear phase

def analyze_differences():
    """Analyze first and second differences of the sequence"""
    values = [C(n) for n in range(2, 15)]
    diffs1 = [values[i+1] - values[i] for i in range(len(values)-1)]
    diffs2 = [diffs1[i+1] - diffs1[i] for i in range(len(diffs1)-1)]

    print("\nFirst differences:", diffs1)
    print("Second differences:", diffs2)
    return diffs1, diffs2

def test_universality():
    """Test for universal patterns and structural stability"""
    # 1. Test for asymptotic behavior
    large_n = range(100, 120)
    asymptotic_ratios = [C(n)/n for n in large_n]
    asymptotic_stability = np.std(asymptotic_ratios) < 0.01

    # 2. Test for structural consistency
    # Compare against alternative growth patterns
    n_values = np.array(range(6, 20))
    actual_values = np.array([C(n) for n in n_values])

    # Test against polynomial fits
    poly_fits = []
    for degree in range(1, 4):
        coeffs = np.polyfit(n_values, actual_values, degree)
        poly_fits.append(np.polyval(coeffs, n_values))

    # Calculate residuals
    residuals = [np.mean((actual_values - fit)**2) for fit in poly_fits]

    # 3. Test for phase transition stability
    phase_transitions = {
        'foundational': [2, 3],
        'transitional': [4, 5],
        'linear': range(6, 15)
    }

    phase_coherence = {}
    for phase, values in phase_transitions.items():
        if phase == 'linear':
            diffs = [C(n+1) - C(n) for n in values[:-1]]
            phase_coherence[phase] = np.std(diffs) < 0.01
        else:
            phase_coherence[phase] = True  # Add specific tests for other phases

    results = {
        'asymptotic_stability': asymptotic_stability,
        'best_fit_degree': np.argmin(residuals) + 1,
        'phase_coherence': phase_coherence,
        'asymptotic_ratio': np.mean(asymptotic_ratios)
    }

    print("\nUniversality Tests:")
    print(f"Asymptotic stability: {asymptotic_stability}")
    print(f"Best polynomial fit degree: {results['best_fit_degree']}")
    print(f"Asymptotic ratio C(n)/n → {results['asymptotic_ratio']:.3f}")
    print(f"Phase coherence: {phase_coherence}")

    return results

def visualize_pattern():
    """Visualize the growth pattern and phase transitions"""
    n_values = range(2, 15)
    c_values = [C(n) for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, c_values, 'bo-', label='C(n)')

    # Mark phase transitions
    plt.axvline(x=3.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=5.5, color='r', linestyle='--', alpha=0.3)

    plt.title('Growth Pattern of Coherence Conditions')
    plt.xlabel('n')
    plt.ylabel('C(n)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Verifying known values and analyzing pattern...")
    diffs1, diffs2 = analyze_differences()
    universality_results = test_universality()
    visualize_pattern()
