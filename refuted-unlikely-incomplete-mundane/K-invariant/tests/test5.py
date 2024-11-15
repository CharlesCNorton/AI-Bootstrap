import numpy as np
import gudhi as gd
import random
from tqdm import tqdm  # Progress bar for multiple iterations

# Define the function to calculate curvature index K_M
def calculate_K_M(c_values, x_values, c_0, exponential=False):
    linear_combination = np.sum(c_values * x_values) + c_0
    quadratic_term = linear_combination ** 2
    sine_term = np.sin(linear_combination)
    if exponential:
        exponential_term = np.exp(0.1 * linear_combination)
        K_M = quadratic_term + sine_term + exponential_term
    else:
        K_M = quadratic_term + sine_term
    return K_M

# Monte Carlo Simulation Parameters
num_tests = 10000
bound_check_results = []

# Statistics to summarize failure conditions
failure_count = 0
dimension_failure_count = {}
vertices_failure_count = {}
complexity_failure_values = []
K_M_failure_values = []
failed_betti_numbers = []

# Monte Carlo Simulations
for test_num in tqdm(range(num_tests)):
    # Step 1: Generate a random simplicial complex with higher dimensionality
    complex = gd.SimplexTree()
    num_vertices = random.randint(10, 25)  # Number of vertices for complexity
    max_dimension = random.randint(2, 6)   # Higher dimensions to test robustness

    # Insert random simplices
    for _ in range(60):  # Insert more simplices to increase density
        simplex = random.sample(range(num_vertices), random.randint(1, max_dimension + 1))
        complex.insert(simplex)

    # Step 2: Compute persistent homology
    complex.compute_persistence()
    betti_numbers = complex.betti_numbers()

    # Calculate homotopical complexity
    complexity = sum(betti_numbers)

    # Step 3: Calculate K_M with systematic parameter variation
    c_values = np.random.uniform(-5, 5, num_vertices)  # Allow negative and positive values
    x_values = np.random.uniform(-10, 15, num_vertices)  # Wider range for x_values, both negative and positive
    c_0 = np.random.uniform(-5, 5)

    K_M = calculate_K_M(c_values, x_values, c_0, exponential=True)

    # Step 4: Check the hypothesis
    constant_factor = 1.5  # Adjust constant factor if needed
    bound_check = K_M >= constant_factor * complexity

    # Record Result
    bound_check_results.append(bound_check)

    # If it fails, record detailed information about the failure
    if not bound_check:
        failure_count += 1
        complexity_failure_values.append(complexity)
        K_M_failure_values.append(K_M)
        failed_betti_numbers.append(betti_numbers)

        # Update failure count by dimension and number of vertices
        if max_dimension in dimension_failure_count:
            dimension_failure_count[max_dimension] += 1
        else:
            dimension_failure_count[max_dimension] = 1

        if num_vertices in vertices_failure_count:
            vertices_failure_count[num_vertices] += 1
        else:
            vertices_failure_count[num_vertices] = 1

        # Print a concise failure statement
        print(f"Failure #{failure_count}: Test {test_num + 1}, Max Dimension: {max_dimension}, "
              f"Vertices: {num_vertices}, Complexity: {complexity}, Betti Numbers: {betti_numbers}, K_M: {K_M}")

# Summary of Results
successful_bounds = sum(bound_check_results)
failure_cases = num_tests - successful_bounds

print("\n=== Monte Carlo Simulation Results ===")
print(f"Total Tests: {num_tests}")
print(f"Successful Bound Checks: {successful_bounds}")
print(f"Failure Cases: {failure_cases}")
print(f"Success Rate: {successful_bounds / num_tests * 100:.2f}%")

if failure_cases > 0:
    print("\n=== Failure Analysis Summary ===")
    # Summary statistics about failure conditions
    print(f"Total Failures: {failure_count}")
    print(f"Dimensions of Complexes Leading to Failures:")
    for dimension, count in dimension_failure_count.items():
        print(f"  Dimension {dimension}: {count} failures")

    print(f"Number of Vertices in Failed Complexes:")
    for vertices, count in vertices_failure_count.items():
        print(f"  {vertices} vertices: {count} failures")

    # Statistical information about K_M values in failure cases
    max_K_M_failure = np.max(K_M_failure_values)
    min_K_M_failure = np.min(K_M_failure_values)
    avg_K_M_failure = np.mean(K_M_failure_values)
    std_K_M_failure = np.std(K_M_failure_values)

    print("\n=== K_M Failure Statistics ===")
    print(f"Maximum K_M in Failures: {max_K_M_failure:.2f}")
    print(f"Minimum K_M in Failures: {min_K_M_failure:.2f}")
    print(f"Average K_M in Failures: {avg_K_M_failure:.2f}")
    print(f"Standard Deviation of K_M in Failures: {std_K_M_failure:.2f}")

    # Complexity statistics in failure cases
    avg_complexity_failure = np.mean(complexity_failure_values) if complexity_failure_values else 0
    max_complexity_failure = np.max(complexity_failure_values) if complexity_failure_values else 0

    print("\n=== Complexity in Failures ===")
    print(f"Average Complexity Value in Failures: {avg_complexity_failure:.2f}")
    print(f"Maximum Complexity Value in Failures: {max_complexity_failure:.2f}")

    # Example Betti Numbers from Failures
    print("\nExample Betti Numbers from Failures (up to 5 samples):")
    print(failed_betti_numbers[:5])
else:
    print("\nSUCCESS: K_M consistently provided an upper bound across all tests.")
