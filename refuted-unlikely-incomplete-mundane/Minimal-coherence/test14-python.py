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

def advanced_analysis():
    """Comprehensive analysis suite"""

    # Generate substantial data
    n_values = range(2, 50)
    c_values = [C(n) for n in n_values]

    results = defaultdict(dict)

    # 1. Growth Rate Analysis
    def analyze_growth():
        ratios = [C(n+1)/C(n) for n in range(2, 49)]
        growth_rates = {
            'early': np.mean(ratios[:4]),
            'transition': np.mean(ratios[4:8]),
            'stable': np.mean(ratios[8:])
        }
        return growth_rates, ratios

    # 2. Pattern Detection
    def detect_patterns():
        sequences = {
            'values': c_values,
            'diffs1': np.diff(c_values),
            'diffs2': np.diff(np.diff(c_values))
        }

        # Test for periodicity
        for name, seq in sequences.items():
            autocorr = np.correlate(seq, seq, mode='full')
            results['patterns'][name] = {
                'autocorr_peaks': find_peaks(autocorr),
                'regularity': np.std(seq)/np.mean(seq)
            }
        return sequences

    # 3. Statistical Tests
    def statistical_analysis():
        # Test linearity in stable phase
        stable_phase = c_values[8:]
        stable_x = range(len(stable_phase))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            stable_x, stable_phase
        )

        # Test for structural breaks
        breaks = find_structural_breaks(c_values)

        return {
            'linear_fit': {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value
            },
            'breaks': breaks
        }

    # 4. Predictive Testing
    def predictive_analysis():
        # Split data and test predictions
        train = c_values[:-5]
        test = c_values[-5:]
        train_x = np.arange(len(train))
        test_x = np.arange(len(train), len(c_values))

        # Fit various models
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

        return predictions

    # 5. Phase Transition Analysis
    def analyze_transitions():
        diffs = np.diff(c_values)
        changes = np.diff(diffs)

        transition_points = []
        for i, change in enumerate(changes):
            if abs(change) > 0.1:
                transition_points.append(i + 3)

        return {
            'points': transition_points,
            'stability': {
                'pre_first': np.std(diffs[:transition_points[0]-2]) if transition_points else np.nan,
                'between': np.std(diffs[transition_points[0]-2:transition_points[-1]-2]) if len(transition_points) > 1 else np.nan,
                'post_last': np.std(diffs[transition_points[-1]-2:]) if transition_points else np.nan
            }
        }

    # Run all analyses
    results['growth'] = analyze_growth()
    results['patterns'] = detect_patterns()
    results['stats'] = statistical_analysis()
    results['predictions'] = predictive_analysis()
    results['transitions'] = analyze_transitions()

    # Print key findings
    print("\nKey Findings:")
    print(f"1. Growth rate stabilizes at: {results['growth'][0]['stable']:.3f}")
    print(f"2. Linear fit R² in stable phase: {results['stats']['linear_fit']['r_squared']:.6f}")
    print(f"3. Number of significant structural breaks: {len(results['transitions']['points'])}")
    print("\nPredictive Accuracy:")
    for model, res in results['predictions'].items():
        if 'mse' in res:
            print(f"{model}: MSE = {res['mse']:.6f}")

    return results

def find_peaks(arr):
    """Helper function to find peaks in array"""
    return [i for i in range(1, len(arr)-1) if arr[i-1] < arr[i] > arr[i+1]]

def find_structural_breaks(data):
    """Detect structural breaks using Chow test approach"""
    breaks = []
    n = len(data)
    for i in range(3, n-3):
        before = data[:i]
        after = data[i:]
        if len(before) > 2 and len(after) > 2:
            t_stat, p_value = stats.ttest_ind(before, after)
            if p_value < 0.05:
                breaks.append(i)
    return breaks

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = advanced_analysis()
