# Import necessary libraries
import random

# Greedy Algorithm in Discrete Setting
def greedy_algorithm(n):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if red_sum < blue_sum:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return red_sum, blue_sum

# Weighted Averaging Algorithm in Discrete Setting
def weighted_averaging_discrete(n, red_weight=0.5):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        if random.random() < red_weight:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return red_sum, blue_sum

# Greedy Algorithm in Continuous Setting
def greedy_algorithm_continuous(n):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        # Fractionally assign based on current difference
        if red_sum < blue_sum:
            red_sum += reciprocal
        else:
            blue_sum += reciprocal
    return red_sum, blue_sum

# Weighted Averaging Algorithm in Continuous Setting
def weighted_averaging_continuous(n, red_weight=0.5):
    red_sum = 0.0
    blue_sum = 0.0
    for i in range(2, n + 2):
        reciprocal = 1 / i
        # Fractionally assign based on weights
        red_sum += red_weight * reciprocal
        blue_sum += (1 - red_weight) * reciprocal
    return red_sum, blue_sum

# Set the value of n for testing
n = 100000000

# Run Greedy Algorithm in Discrete Setting
greedy_discrete_red_sum, greedy_discrete_blue_sum = greedy_algorithm(n)
greedy_discrete_discrepancy = abs(greedy_discrete_red_sum - greedy_discrete_blue_sum)

# Run Weighted Averaging Algorithm in Discrete Setting (50/50 Split)
weighted_discrete_red_sum, weighted_discrete_blue_sum = weighted_averaging_discrete(n, red_weight=0.5)
weighted_discrete_discrepancy = abs(weighted_discrete_red_sum - weighted_discrete_blue_sum)

# Run Greedy Algorithm in Continuous Setting
greedy_continuous_red_sum, greedy_continuous_blue_sum = greedy_algorithm_continuous(n)
greedy_continuous_discrepancy = abs(greedy_continuous_red_sum - greedy_continuous_blue_sum)

# Run Weighted Averaging Algorithm in Continuous Setting (50/50 Split)
weighted_continuous_red_sum, weighted_continuous_blue_sum = weighted_averaging_continuous(n)
weighted_continuous_discrepancy = abs(weighted_continuous_red_sum - weighted_continuous_blue_sum)

# Output results for all four test cases
results = {
    "Greedy_Algorithm_Discrete": {
        "Red_Sum": greedy_discrete_red_sum,
        "Blue_Sum": greedy_discrete_blue_sum,
        "Discrepancy": greedy_discrete_discrepancy
    },
    "Weighted_Averaging_Discrete": {
        "Red_Sum": weighted_discrete_red_sum,
        "Blue_Sum": weighted_discrete_blue_sum,
        "Discrepancy": weighted_discrete_discrepancy
    },
    "Greedy_Algorithm_Continuous": {
        "Red_Sum": greedy_continuous_red_sum,
        "Blue_Sum": greedy_continuous_blue_sum,
        "Discrepancy": greedy_continuous_discrepancy
    },
    "Weighted_Averaging_Continuous": {
        "Red_Sum": weighted_continuous_red_sum,
        "Blue_Sum": weighted_continuous_blue_sum,
        "Discrepancy": weighted_continuous_discrepancy
    }
}

# Print the results
print(results)
