import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from collections import defaultdict
from tqdm import tqdm

# Configure logging for unstable cases
logging.basicConfig(filename='unstable_cases.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
A0 = 1.0  # Base point a0
EPSILON_START = -3.0
EPSILON_END = 3.0
EPSILON_STEP = 0.005
DELTA = 2.0  # Stability threshold
HOMOTOPY_LEVEL_START = 1
HOMOTOPY_LEVEL_END = 100

# Generate perturbation values
EPSILONS = np.arange(EPSILON_START, EPSILON_END + EPSILON_STEP, EPSILON_STEP)

# Function Definitions

def P(a0, epsilon):
    """Perturbation Function P(a0, epsilon) = a0 + epsilon"""
    return a0 + epsilon

def dynamic_cup_product(n, epsilon, fiber_number):
    """
    Dynamic calculation of cup products based on homotopy level and perturbation.
    Example definition: cup_product = sin(n * epsilon) + fiber_number
    """
    return math.sin(n * epsilon) + fiber_number

def L_n(a0, epsilon, n):
    """Loop Space Type L_n(a0, epsilon)"""
    perturbation = P(a0, epsilon)
    try:
        term1 = ((a0 + perturbation) / 2) ** (1 / n)
    except (ZeroDivisionError, ValueError):
        term1 = np.nan  # Handle invalid operations
    term2 = math.cos(n * (a0 + epsilon))
    return term1 + term2

def P_n(a0_1, a0_2, epsilon, n):
    """Product Type P_n(a0^1, a0^2, epsilon)"""
    try:
        term1 = (a0_1 + epsilon) ** (1 / n) + math.cos(n * (a0_1 + epsilon))
    except (ZeroDivisionError, ValueError):
        term1 = np.nan
    try:
        term2 = (a0_2 - epsilon) ** (1 / n) + math.sin(n * (a0_2 - epsilon))
    except (ZeroDivisionError, ValueError):
        term2 = np.nan
    return (term1 + term2) / 2

def F_n(a0_base, a0_fiber1, a0_fiber2, epsilon, n):
    """Fibration Type F_n(a0_base, a0_fiber1, a0_fiber2, epsilon)"""
    try:
        base_term = (a0_base + epsilon) ** (1 / n) + math.cos(n * a0_base)
    except (ZeroDivisionError, ValueError):
        base_term = np.nan

    # Dynamic cup products based on n and epsilon
    cup_product1 = dynamic_cup_product(n, epsilon, fiber_number=1)
    cup_product2 = dynamic_cup_product(n, epsilon, fiber_number=2)

    try:
        fiber1_term = (a0_fiber1 + 0.5 * epsilon) ** (1 / (n + 1)) + math.sin(n * a0_fiber1) + cup_product1
    except (ZeroDivisionError, ValueError):
        fiber1_term = np.nan
    fiber1_term /= 2

    try:
        fiber2_term = (a0_fiber2 + 0.25 * epsilon) ** (1 / (n + 2)) + math.sin(n * a0_fiber2) + cup_product2
    except (ZeroDivisionError, ValueError):
        fiber2_term = np.nan
    fiber2_term /= 2

    try:
        return (base_term + fiber1_term + fiber2_term) / 2
    except TypeError:
        return np.nan

def evaluate_stability(value, delta):
    """Check if the absolute value is less than delta"""
    if np.isnan(value):
        return False
    return abs(value) < delta

# Data Structures for Statistics
statistics = {
    'Loop Space': [],
    'Product Type': [],
    'Fibration Type': []
}

unstable_cases = defaultdict(list)

# Main Evaluation Loop
def main():
    print("Starting Comprehensive Stability Evaluation for Higher Homotopies...\n")

    # Iterate through homotopy levels with progress bar
    for n in tqdm(range(HOMOTOPY_LEVEL_START, HOMOTOPY_LEVEL_END + 1), desc="Homotopy Levels"):
        for epsilon in EPSILONS:
            # Loop Space Evaluation
            l_n = L_n(A0, epsilon, n)
            is_l_n_stable = evaluate_stability(l_n, DELTA)
            statistics['Loop Space'].append({
                'n': n,
                'epsilon': epsilon,
                'value': l_n,
                'stable': is_l_n_stable
            })
            if not is_l_n_stable:
                logging.info(f"Loop Space - n={n}, epsilon={epsilon}, value={l_n}")
                unstable_cases['Loop Space'].append({'n': n, 'epsilon': epsilon, 'value': l_n})

            # Product Type Evaluation
            # Assuming a0^1 and a0^2 are both A0 for simplicity
            p_n = P_n(A0, A0, epsilon, n)
            is_p_n_stable = evaluate_stability(p_n, DELTA)
            statistics['Product Type'].append({
                'n': n,
                'epsilon': epsilon,
                'value': p_n,
                'stable': is_p_n_stable
            })
            if not is_p_n_stable:
                logging.info(f"Product Type - n={n}, epsilon={epsilon}, value={p_n}")
                unstable_cases['Product Type'].append({'n': n, 'epsilon': epsilon, 'value': p_n})

            # Fibration Type Evaluation
            # Assuming a0_base, a0_fiber1, a0_fiber2 are all A0 for simplicity
            f_n = F_n(A0, A0, A0, epsilon, n)
            is_f_n_stable = evaluate_stability(f_n, DELTA)
            statistics['Fibration Type'].append({
                'n': n,
                'epsilon': epsilon,
                'value': f_n,
                'stable': is_f_n_stable
            })
            if not is_f_n_stable:
                logging.info(f"Fibration Type - n={n}, epsilon={epsilon}, value={f_n}")
                unstable_cases['Fibration Type'].append({'n': n, 'epsilon': epsilon, 'value': f_n})

    # Convert statistics to DataFrames for easier analysis
    stats_df = {}
    for key, records in statistics.items():
        stats_df[key] = pd.DataFrame(records)

    # Generate Statistical Summary
    summary = {}
    for key, df in stats_df.items():
        total = len(df)
        stable = df['stable'].sum()
        unstable = total - stable
        stability_ratio = (stable / total) * 100
        mean_value = df['value'].mean()
        std_dev = df['value'].std()
        summary[key] = {
            'Total Evaluations': total,
            'Stable': stable,
            'Unstable': unstable,
            'Stability (%)': stability_ratio,
            'Mean Value': mean_value,
            'Standard Deviation': std_dev
        }

    summary_df = pd.DataFrame(summary).T
    print("\nStability Evaluation Summary:")
    print(summary_df)

    # Detailed Statistics by Homotopy Level
    detailed_stats = {}
    for key, df in stats_df.items():
        detailed_stats[key] = df.groupby('n')['stable'].agg(['sum', 'count'])
        detailed_stats[key]['Stability (%)'] = (detailed_stats[key]['sum'] / detailed_stats[key]['count']) * 100

    print("\nDetailed Stability by Homotopy Level:")
    for key, df in detailed_stats.items():
        print(f"\n{key}:")
        print(df[['sum', 'count', 'Stability (%)']].rename(columns={'sum': 'Stable', 'count': 'Total'}))

    # Visualization
    # Plot Stability Percentage vs Homotopy Level for each type
    plt.figure(figsize=(14, 8))
    for key, df in detailed_stats.items():
        plt.plot(df.index, df['Stability (%)'], marker='o', label=key)
    plt.title('Stability Percentage vs Homotopy Level', fontsize=16)
    plt.xlabel('Homotopy Level (n)', fontsize=14)
    plt.ylabel('Stability (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('stability_percentage_vs_homotopy_level.png')
    plt.show()

    # Plot Distribution of Values for Each Type
    for key, df in stats_df.items():
        plt.figure(figsize=(14, 6))
        plt.hist(df['value'].dropna(), bins=200, alpha=0.7, label=key)
        plt.title(f'Distribution of Values for {key}', fontsize=16)
        plt.xlabel('Function Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(f'distribution_of_values_{key.replace(" ", "_")}.png')
        plt.show()

    # Heatmap of Stability Percentage
    heatmap_data = {}
    for key, df in detailed_stats.items():
        heatmap_data[key] = df['Stability (%)'].values
    heatmap_df = pd.DataFrame(heatmap_data, index=detailed_stats['Loop Space'].index)

    plt.figure(figsize=(14, 20))
    plt.imshow(heatmap_df, cmap='viridis', aspect='auto')
    plt.colorbar(label='Stability (%)')
    plt.title('Heatmap of Stability Percentage by Homotopy Level and Type', fontsize=16)
    plt.xlabel('Homotopy Type', fontsize=14)
    plt.ylabel('Homotopy Level (n)', fontsize=14)
    plt.xticks(ticks=np.arange(len(heatmap_df.columns)), labels=heatmap_df.columns, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(heatmap_df.index)), labels=heatmap_df.index)
    plt.savefig('heatmap_stability_percentage.png')
    plt.show()

    # Save Detailed Statistics to CSV
    for key, df in stats_df.items():
        df.to_csv(f'detailed_statistics_{key.replace(" ", "_")}.csv', index=False)

    # Save Summary to CSV
    summary_df.to_csv('stability_summary.csv')

    # Save Detailed Stability by Homotopy Level to CSV
    for key, df in detailed_stats.items():
        df.to_csv(f'detailed_stability_by_level_{key.replace(" ", "_")}.csv')

    print("\nEvaluation complete. Detailed statistics, visualizations, and logs have been saved.")

if __name__ == "__main__":
    main()
