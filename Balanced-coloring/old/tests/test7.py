    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Greedy Coloring Algorithm
    def greedy_coloring(n):
        S_red, S_blue = 0, 0
        discrepancies = []
        for i in range(2, n + 2):
            reciprocal = 1 / i
            if abs(S_red + reciprocal - S_blue) < abs(S_red - (S_blue + reciprocal)):
                S_red += reciprocal
            else:
                S_blue += reciprocal
            discrepancies.append(S_red - S_blue)
        return discrepancies

    # Ratio-Based Coloring Algorithm
    def ratio_based_coloring(n):
        S_red, S_blue = 0, 0
        discrepancies = []
        for i in range(2, n + 2):
            reciprocal = 1 / i
            if S_red > S_blue:
                S_blue += reciprocal
            else:
                S_red += reciprocal
            discrepancies.append(S_red - S_blue)
        return discrepancies

    # Weighted Averaging Algorithm
    def weighted_averaging_coloring(n):
        S_red, S_blue = 0, 0
        discrepancies = []
        for i in range(2, n + 2):
            reciprocal = 1 / i
            if S_red < S_blue:
                S_red += reciprocal * 0.6
                S_blue += reciprocal * 0.4
            else:
                S_red += reciprocal * 0.4
                S_blue += reciprocal * 0.6
            discrepancies.append(S_red - S_blue)
        return discrepancies

    # Function to measure execution time and final discrepancy
    def run_torture_test(algorithm, n):
        start_time = time.time()
        discrepancies = algorithm(n)
        end_time = time.time()
        execution_time = end_time - start_time
        final_discrepancy = discrepancies[-1]
        return execution_time, final_discrepancy

    # Compare algorithms
    def compare_algorithms(n):
        algorithms = {
            "Greedy Coloring": greedy_coloring,
            "Ratio-Based Coloring": ratio_based_coloring,
            "Weighted Averaging": weighted_averaging_coloring
        }

        results = []
        for name, algorithm in algorithms.items():
            execution_time, final_discrepancy = run_torture_test(algorithm, n)
            results.append({"Algorithm": name, "Execution Time (s)": execution_time, "Final Discrepancy": final_discrepancy})

        # Display results
        results_df = pd.DataFrame(results)
        print(results_df)

        # Plot discrepancies for each algorithm
        plt.figure(figsize=(10, 6))
        for name, algorithm in algorithms.items():
            discrepancies = algorithm(n)
            plt.plot(discrepancies, label=name)
        plt.title(f"Discrepancy Evolution for n={n}")
        plt.xlabel("Step (n)")
        plt.ylabel("Discrepancy (S_red - S_blue)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Run the comparison for a specified n
    n = 1000000000  # You can adjust this value for larger torture tests
    compare_algorithms(n)
