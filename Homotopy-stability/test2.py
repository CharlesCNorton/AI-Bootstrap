import math
import numpy as np

# Constants
A0 = 1.0  # Base point a0
EPSILONS = np.linspace(-0.5, 0.5, 11)  # Perturbations from -0.5 to 0.5 in steps of 0.1
DELTA = 2.0  # Stability threshold

# Placeholder cup products
CUP_PRODUCT_1 = 1.0
CUP_PRODUCT_2 = 1.0
# Add more cup products as needed

# Function Definitions

def P(a0, epsilon):
    """Perturbation Function P(a0, epsilon) = a0 + epsilon"""
    return a0 + epsilon

def L_n(a0, epsilon, n):
    """Loop Space Type L_n(a0, epsilon)"""
    perturbation = P(a0, epsilon)
    term1 = ((a0 + perturbation) / 2) ** (1 / n)
    term2 = math.cos(n * (a0 + epsilon))
    return term1 + term2

def P_n(a0_1, a0_2, epsilon, n):
    """Product Type P_n(a0^1, a0^2, epsilon)"""
    term1 = (a0_1 + epsilon) ** (1 / n) + math.cos(n * (a0_1 + epsilon))
    term2 = (a0_2 - epsilon) ** (1 / n) + math.sin(n * (a0_2 - epsilon))
    return (term1 + term2) / 2

def F_n(a0_base, a0_fiber1, a0_fiber2, epsilon, n):
    """Fibration Type F_n(a0_base, a0_fiber1, a0_fiber2, epsilon)"""
    base_term = (a0_base + epsilon) ** (1 / n) + math.cos(n * a0_base)

    fiber1_term = (a0_fiber1 + 0.5 * epsilon) ** (1 / (n + 1)) + math.sin(n * a0_fiber1) + CUP_PRODUCT_1
    fiber1_term /= 2

    fiber2_term = (a0_fiber2 + 0.25 * epsilon) ** (1 / (n + 2)) + math.sin(n * a0_fiber2) + CUP_PRODUCT_2
    fiber2_term /= 2

    return (base_term + fiber1_term + fiber2_term) / 2

def evaluate_stability(value, delta):
    """Check if the absolute value is less than delta"""
    return abs(value) < delta

# Data Structures for Statistics
statistics = {
    'Loop Space': {'stable': 0, 'total': 0},
    'Product Type': {'stable': 0, 'total': 0},
    'Fibration Type': {'stable': 0, 'total': 0}
}

# Main Evaluation Loop
def main():
    print("Starting Stability Evaluation for Higher Homotopies...\n")

    for n in range(1, 6):  # Homotopy levels 1 to 5
        print(f"Evaluating Homotopy Level n = {n}")
        for epsilon in EPSILONS:
            # Loop Space Evaluation
            l_n = L_n(A0, epsilon, n)
            is_l_n_stable = evaluate_stability(l_n, DELTA)
            statistics['Loop Space']['total'] += 1
            if is_l_n_stable:
                statistics['Loop Space']['stable'] += 1

            # Product Type Evaluation
            # Assuming a0^1 and a0^2 are both A0 for simplicity
            p_n = P_n(A0, A0, epsilon, n)
            is_p_n_stable = evaluate_stability(p_n, DELTA)
            statistics['Product Type']['total'] += 1
            if is_p_n_stable:
                statistics['Product Type']['stable'] += 1

            # Fibration Type Evaluation
            # Assuming a0_base, a0_fiber1, a0_fiber2 are all A0 for simplicity
            f_n = F_n(A0, A0, A0, epsilon, n)
            is_f_n_stable = evaluate_stability(f_n, DELTA)
            statistics['Fibration Type']['total'] += 1
            if is_f_n_stable:
                statistics['Fibration Type']['stable'] += 1

        print(f"Completed evaluations for n = {n}\n")

    # Statistical Summary
    print("Stability Evaluation Summary:")
    for key, value in statistics.items():
        stability_ratio = (value['stable'] / value['total']) * 100
        print(f"- {key}: {value['stable']} stable out of {value['total']} evaluations ({stability_ratio:.2f}%)")

    # Additional Detailed Statistics (Optional)
    # This section can be expanded to include more granular statistics if needed.

if __name__ == "__main__":
    main()
