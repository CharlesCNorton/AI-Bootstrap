import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import itertools
import logging

# Configure logging for detailed debug information
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define constants
DELTA_BASE = 2.0  # Base stability threshold
DELTA_FACTOR = 1.0  # Factor to adjust with level

# Define perturbation ranges
EPSILON_VALUES = np.linspace(-1.0, 1.0, 100)  # 100 values from -1 to 1
A0_VALUES = np.linspace(0.1, 10.0, 100)       # 100 values from 0.1 to 10 to avoid division by zero

# Homotopy levels to test
HOMOTOPY_LEVELS = range(1, 6)  # Levels 1 through 5

# Define cup products based on cohomological interactions
def cup_product_1(n, a0_fiber1):
    # Example: Modulated by n and a0_fiber1
    return np.sin(n * a0_fiber1) * 0.5

def cup_product_2(n, a0_fiber2):
    return np.cos(n * a0_fiber2) * 0.5

# Adaptive scaling function
def adaptive_scaling(n):
    return 1 / (1 + 0.5 * n)  # Adjusted scaling factor

# Perturbation function
def P(a0, epsilon):
    return a0 + epsilon

# Loop Space calculation
def loop_space(n, a0, epsilon, scaling_factor):
    scaled_epsilon = epsilon * scaling_factor
    perturbed = P(a0, scaled_epsilon)
    try:
        term1 = ((a0 + perturbed) / 2) ** (1 / n)
    except ZeroDivisionError:
        term1 = 0
    term2 = np.cos(n * (a0 + scaled_epsilon))
    result = term1 + term2
    logging.debug(f"L_{n}({a0}, {epsilon}) = {result}")
    return result

# Product Type calculation
def product_type(n, a0_1, a0_2, epsilon, scaling_factor):
    scaled_epsilon = epsilon * scaling_factor
    perturbed1 = (a0_1 + scaled_epsilon) ** (1 / n)
    term1 = perturbed1 + np.cos(n * (a0_1 + scaled_epsilon))

    perturbed2 = (a0_2 - scaled_epsilon) ** (1 / n)
    term2 = perturbed2 + np.sin(n * (a0_2 - scaled_epsilon))

    result = (term1 + term2) / 2
    logging.debug(f"P_{n}({a0_1}, {a0_2}, {epsilon}) = {result}")
    return result

# Fibration Type calculation
def fibration_type(n, a0_base, a0_fiber1, a0_fiber2, epsilon, scaling_factor):
    scaled_epsilon = epsilon * scaling_factor
    base_perturbed = (a0_base + scaled_epsilon) ** (1 / n)
    base_term = base_perturbed + np.cos(n * a0_base)

    fiber1_perturbed = (a0_fiber1 + 0.5 * scaled_epsilon) ** (1 / (n + 1))
    fiber1_term = fiber1_perturbed + np.sin(n * a0_fiber1) + cup_product_1(n, a0_fiber1)
    fiber1_avg = fiber1_term / 2

    fiber2_perturbed = (a0_fiber2 + 0.25 * scaled_epsilon) ** (1 / (n + 2))
    fiber2_term = fiber2_perturbed + np.sin(n * a0_fiber2) + cup_product_2(n, a0_fiber2)
    fiber2_avg = fiber2_term / 2

    result = (base_term + fiber1_avg + fiber2_avg) / 2
    logging.debug(f"F_{n}({a0_base}, {a0_fiber1}, {a0_fiber2}, {epsilon}) = {result}")
    return result

# Stability condition
def is_stable(value, delta):
    return abs(value) < delta

# Main evaluation function
def evaluate_homotopies():
    # Initialize data storage
    results = {
        'Homotopy_Type': [],
        'Level': [],
        'Perturbation': [],
        'Stable': [],
        'Value': []
    }

    # Iterate over homotopy levels
    for n in HOMOTOPY_LEVELS:
        scaling_factor = adaptive_scaling(n)
        delta_n = DELTA_BASE * DELTA_FACTOR * (1 + n)  # Example scaling
        logging.info(f"Evaluating Homotopy Level {n} with Scaling Factor {scaling_factor:.4f} and Delta {delta_n}")

        # Iterate over epsilon and a0 values
        for epsilon, a0 in itertools.product(EPSILON_VALUES, A0_VALUES):
            # Loop Space Evaluation
            L_n = loop_space(n, a0, epsilon, scaling_factor)
            stable_L = is_stable(L_n, delta_n)
            results['Homotopy_Type'].append('Loop Space')
            results['Level'].append(n)
            results['Perturbation'].append(epsilon)
            results['Stable'].append(stable_L)
            results['Value'].append(L_n)

            # Product Type Evaluation
            # For Product Type, we need two a0 values; using the same a0 for simplicity
            P_n = product_type(n, a0, a0, epsilon, scaling_factor)
            stable_P = is_stable(P_n, delta_n)
            results['Homotopy_Type'].append('Product Type')
            results['Level'].append(n)
            results['Perturbation'].append(epsilon)
            results['Stable'].append(stable_P)
            results['Value'].append(P_n)

            # Fibration Type Evaluation
            # For Fibration Type, we need base and two fiber a0 values; using the same a0 for simplicity
            F_n = fibration_type(n, a0, a0, a0, epsilon, scaling_factor)
            stable_F = is_stable(F_n, delta_n)
            results['Homotopy_Type'].append('Fibration Type')
            results['Level'].append(n)
            results['Perturbation'].append(epsilon)
            results['Stable'].append(stable_F)
            results['Value'].append(F_n)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    return df_results

# Statistical analysis function
def analyze_results(df):
    # Initialize summary storage
    summary = {
        'Homotopy_Type': [],
        'Level': [],
        'Total_Evaluations': [],
        'Stable': [],
        'Unstable': [],
        'Stability_Percentage': [],
        'Mean_Value': [],
        'Std_Dev': []
    }

    # Group by Homotopy_Type and Level
    grouped = df.groupby(['Homotopy_Type', 'Level'])

    for (homotopy_type, level), group in grouped:
        total = len(group)
        stable = group['Stable'].sum()
        unstable = total - stable
        stability_pct = (stable / total) * 100
        mean_val = group['Value'].mean()
        std_dev = group['Value'].std()

        summary['Homotopy_Type'].append(homotopy_type)
        summary['Level'].append(level)
        summary['Total_Evaluations'].append(total)
        summary['Stable'].append(stable)
        summary['Unstable'].append(unstable)
        summary['Stability_Percentage'].append(round(stability_pct, 2))
        summary['Mean_Value'].append(round(mean_val, 4))
        summary['Std_Dev'].append(round(std_dev, 4))

    # Convert summary to DataFrame
    df_summary = pd.DataFrame(summary)
    return df_summary

# Function to display detailed statistics
def detailed_statistics(df):
    stats = df.describe()
    return stats

# Function to save results to CSV
def save_results(df, filename):
    df.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")

# Function to generate summary tables
def generate_summary_tables(df_summary):
    for homotopy_type in df_summary['Homotopy_Type'].unique():
        subset = df_summary[df_summary['Homotopy_Type'] == homotopy_type]
        print(f"\n### {homotopy_type} Stability Summary\n")
        print(subset.to_string(index=False))
    print("\n### Overall Stability Summary\n")
    print(df_summary.to_string(index=False))

# Main execution
def main():
    logging.info("Starting Homotopy Stability Evaluation Program")

    # Evaluate homotopies
    df_results = evaluate_homotopies()
    logging.info("Completed Homotopy Evaluations")

    # Analyze results
    df_summary = analyze_results(df_results)
    logging.info("Completed Statistical Analysis")

    # Generate summary tables
    generate_summary_tables(df_summary)

    # Optionally, save results to CSV
    # save_results(df_results, 'homotopy_results.csv')
    # save_results(df_summary, 'homotopy_summary.csv')

    logging.info("Program Completed Successfully")

if __name__ == "__main__":
    main()
