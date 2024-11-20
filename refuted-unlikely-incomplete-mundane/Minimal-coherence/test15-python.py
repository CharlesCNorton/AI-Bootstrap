import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict

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

def detailed_phase_analysis():
    """Analyze specific properties of each phase"""
    phases = {
        'foundational': range(2, 4),
        'transitional': range(4, 6),
        'stable': range(6, 20)
    }

    phase_metrics = {}
    for phase_name, phase_range in phases.items():
        values = [C(n) for n in phase_range]
        diffs = np.diff(values)

        phase_metrics[phase_name] = {
            'mean_increment': np.mean(diffs),
            'increment_stability': np.std(diffs),
            'local_linearity': stats.linregress(list(phase_range), values).rvalue**2,
            'values': values
        }

    return phase_metrics

def test_algebraic_properties():
    """Test for underlying algebraic structure"""
    n_range = range(2, 15)
    values = [C(n) for n in n_range]

    # Test for arithmetic sequence in stable phase
    stable_values = values[4:]
    differences = np.diff(stable_values)
    is_arithmetic = np.allclose(differences, differences[0])

    # Test for geometric properties
    ratios = [C(n+1)/C(n) for n in n_range[:-1]]
    ratio_convergence = np.std(ratios[4:])

    return {
        'is_arithmetic_sequence': is_arithmetic,
        'ratio_convergence': ratio_convergence,
        'stable_difference': differences[0] if is_arithmetic else None
    }

def advanced_analysis():
    """Comprehensive analysis suite"""

    # Generate substantial data
    n_values = range(2, 50)
    c_values = [C(n) for n in n_values]

    results = defaultdict(dict)

    # 1. Growth Rate Analysis
    ratios = [C(n+1)/C(n) for n in range(2, 49)]
    growth_rates = {
        'early': np.mean(ratios[:4]),
        'transition': np.mean(ratios[4:8]),
        'stable': np.mean(ratios[8:])
    }

    # 2. Pattern Analysis
    diffs1 = np.diff(c_values)
    diffs2 = np.diff(diffs1)

    pattern_stats = {
        'main_sequence': {
            'mean': np.mean(c_values),
            'std': np.std(c_values),
            'growth_rate': np.mean(ratios)
        },
        'first_differences': {
            'mean': np.mean(diffs1),
            'std': np.std(diffs1)
        },
        'second_differences': {
            'mean': np.mean(diffs2),
            'std': np.std(diffs2)
        }
    }

    # 3. Statistical Tests
    stable_phase = c_values[8:]
    stable_x = range(len(stable_phase))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        stable_x, stable_phase
    )

    # 4. Predictive Testing
    train = c_values[:-5]
    test = c_values[-5:]
    train_x = np.arange(len(train))
    test_x = np.arange(len(train), len(c_values))

    models = {
        'linear': lambda x, a, b: a*x + b,
        'quadratic': lambda x, a, b, c: a*x**2 + b*x + c,
        'exponential': lambda x, a, b: a*np.exp(b*x)
    }

    predictions = {}
    for name, model in models.items():
        try:
            popt, _ = curve_fit(model, train_x, train)
            pred = model(test_x, *popt)
            predictions[name] = {
                'params': popt,
                'mse': np.mean((pred - test)**2)
            }
        except:
            predictions[name] = {'error': 'Fit failed'}

    # 5. Additional Analysis
    phase_metrics = detailed_phase_analysis()
    algebraic_props = test_algebraic_properties()

    # Print Results
    print("\nKey Findings:")
    print(f"1. Growth rate stabilizes at: {growth_rates['stable']:.3f}")
    print(f"2. Linear fit R² in stable phase: {r_value**2:.6f}")
    print(f"3. Number of significant phase transitions: {len(phase_metrics)}")

    print("\nPredictive Accuracy:")
    for model, res in predictions.items():
        if 'mse' in res:
            print(f"{model}: MSE = {res['mse']:.6f}")

    print("\nPhase Analysis:")
    for phase, metrics in phase_metrics.items():
        print(f"\n{phase.capitalize()} Phase:")
        print(f"Mean increment: {metrics['mean_increment']:.3f}")
        print(f"Increment stability: {metrics['increment_stability']:.3f}")
        print(f"Local linearity (R²): {metrics['local_linearity']:.3f}")

    print("\nAlgebraic Properties:")
    print(f"Is arithmetic sequence in stable phase: {algebraic_props['is_arithmetic_sequence']}")
    print(f"Ratio convergence (std): {algebraic_props['ratio_convergence']:.6f}")
    print(f"Stable difference: {algebraic_props['stable_difference']}")

    return {
        'growth_rates': growth_rates,
        'pattern_stats': pattern_stats,
        'predictions': predictions,
        'phase_metrics': phase_metrics,
        'algebraic_properties': algebraic_props
    }

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = advanced_analysis()
