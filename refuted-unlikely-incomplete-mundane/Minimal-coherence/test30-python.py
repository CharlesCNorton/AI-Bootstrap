# Import necessary libraries
import numpy as np
import gudhi as gd
from gudhi import CubicalComplex
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to compute minimal coherence conditions C(n)
def C(n):
    if n >= 2:
        if n in [2, 3]:
            return n - 1
        elif n in [4, 5]:
            return 2 * n - 3
        elif n >= 6:
            return 2 * n - 1
    else:
        raise ValueError("n must be an integer greater than or equal to 2.")

# Generate C(n) for n from 2 to 50
n_values = np.arange(2, 51)
C_values = np.array([C(n) for n in n_values])

# Phase Transition Analysis
def phase_transition_analysis(n_vals, C_vals):
    # Calculate first and second differences
    first_diff = np.diff(C_vals)
    second_diff = np.diff(first_diff)

    # Identify phase transition points
    transitions = []
    for i in range(1, len(second_diff)):
        if second_diff[i] != second_diff[i-1]:
            transitions.append(n_vals[i+1])

    return transitions

transitions = phase_transition_analysis(n_values, C_values)

# Topological Data Analysis using GUDHI
def topological_data_analysis(C_vals):
    # Normalize data for cubical complex
    scaler = MinMaxScaler()
    C_normalized = scaler.fit_transform(C_vals.reshape(-1, 1)).flatten()

    # Create a cubical complex
    cubical_complex = CubicalComplex(dimensions=[len(C_normalized)], top_dimensional_cells=C_normalized)

    # Compute persistence diagram
    persistence = cubical_complex.persistence()

    # Calculate Betti numbers
    betti_numbers = cubical_complex.betti_numbers()

    return persistence, betti_numbers

persistence, betti_numbers = topological_data_analysis(C_values)

# Fractal Dimension Calculation
def fractal_dimension(C_vals):
    # Calculate the correlation dimension using the Grassberger-Procaccia algorithm
    distances = pdist(C_vals.reshape(-1, 1))
    distances = distances[distances > 0]  # Remove zero distances
    distances.sort()
    n = len(C_vals)
    r_values = np.logspace(np.log10(distances.min()), np.log10(distances.max()), num=50)
    C_r = []
    for r in r_values:
        count = np.sum(distances < r)
        C_r.append(count / (n * (n - 1) / 2))
    # Fit line to log-log plot
    log_r = np.log(r_values)
    log_C_r = np.log(C_r)
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_C_r)
    return slope  # Fractal dimension estimate

frac_dim_estimate = fractal_dimension(C_values)

# Detailed Statistical Analysis
def statistical_analysis(n_vals, C_vals):
    # Perform linear regression for each phase
    phases = []
    start_idx = 0
    for idx in range(1, len(n_vals)):
        if n_vals[idx] in transitions or idx == len(n_vals) - 1:
            end_idx = idx
            n_phase = n_vals[start_idx:end_idx + 1]
            C_phase = C_vals[start_idx:end_idx + 1]
            slope, intercept, r_value, p_value, std_err = linregress(n_phase, C_phase)
            phases.append({
                'start_n': n_phase[0],
                'end_n': n_phase[-1],
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err
            })
            start_idx = end_idx + 1
    return phases

phases_stats = statistical_analysis(n_values, C_values)

# Output all results without simplifications or placeholders
def output_results():
    print("Phase Transitions Detected at n values:", transitions)
    print("\nStatistical Analysis for Each Phase:")
    for phase in phases_stats:
        print(f"n from {phase['start_n']} to {phase['end_n']}:")
        print(f"  Slope = {phase['slope']}")
        print(f"  Intercept = {phase['intercept']}")
        print(f"  R-squared = {phase['r_squared']}")
        print(f"  p-value = {phase['p_value']}")
        print(f"  Standard Error = {phase['std_err']}\n")
    print("Topological Data Analysis:")
    print(f"  Betti Numbers: {betti_numbers}")
    print(f"  Persistence Diagram: {persistence}")
    print("\nFractal Dimension Estimate:")
    print(f"  Fractal Dimension = {frac_dim_estimate}")

# Main execution
if __name__ == "__main__":
    output_results()

    # Optional: Plotting (Commented out since no sample output is required)
    # plt.figure()
    # plt.plot(n_values, C_values, marker='o')
    # plt.xlabel('n')
    # plt.ylabel('C(n)')
    # plt.title('Minimal Coherence Conditions C(n) vs. n')
    # plt.grid(True)
    # plt.show()
