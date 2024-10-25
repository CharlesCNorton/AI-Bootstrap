from math import isqrt
from collections import defaultdict

# Define the attractor ratios for comparison
attractors = {
    'α': (0.75, 0.8, 0.6),
    'β': (1.333, 0.6, 0.8),
    'γ': (0.417, 0.923, 0.385)
}

def generate_solutions(limit):
    solutions = []
    for z in range(2, limit):
        for x in range(1, z):
            y_squared = z**2 + 1 - x**2
            if y_squared > 0:
                y = isqrt(y_squared)
                if y**2 == y_squared:
                    solutions.append((x, y, z))
    return solutions

# Function to classify ratios into attractor or non-attractor categories
def classify_ratios(solutions, attractors, tolerance=0.05):
    attractor_counts = {key: 0 for key in attractors.keys()}
    non_attractor_count = 0

    for x, y, z in solutions:
        # Compute the ratios
        ratios = (x / y, y / z, x / z)

        matched = False
        # Compare with each attractor
        for key, attractor in attractors.items():
            if all(abs(ratio - attractor_val) <= tolerance for ratio, attractor_val in zip(ratios, attractor)):
                attractor_counts[key] += 1
                matched = True
                break

        if not matched:
            non_attractor_count += 1

    return attractor_counts, non_attractor_count

# Main function to run the test stepwise from 10,000 to 100,000 with a step size of 10,000
def main():
    step_values = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    for limit in step_values:
        print(f"\nGenerating solutions for z up to {limit}...\n")
        solutions = generate_solutions(limit)
        attractor_counts, non_attractor_count = classify_ratios(solutions, attractors)

        # Displaying the counts of attractor vs non-attractor solutions
        print(f"Results for z < {limit}:")
        for key, count in attractor_counts.items():
            print(f"Attractor {key}: {count} solutions")
        print(f"Non-attractor: {non_attractor_count} solutions")

if __name__ == "__main__":
    main()

