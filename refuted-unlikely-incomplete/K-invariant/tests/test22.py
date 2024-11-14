import numpy as np
import gudhi as gd
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, KFold
import optuna
from deap import base, creator, tools, algorithms
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ================================
# Configuration and Parameters
# ================================

# Initial constants for the invariant K_M
INITIAL_CONSTANT_FACTOR = 1.5
INITIAL_INTERACTION_STRENGTH = 0.5

# Parameter grid for grid search optimization
PARAMETER_GRID = {
    'CONSTANT_FACTOR': np.arange(1.0, 3.1, 0.1),       # From 1.0 to 3.0 with step 0.1
    'INTERACTION_STRENGTH': np.arange(0.1, 1.1, 0.1)   # From 0.1 to 1.0 with step 0.1
}

# Define constants for Bayesian Optimization
OPTUNA_SAMPLER = optuna.samplers.TPESampler(seed=42)

# Define constants for Genetic Algorithms
POPULATION_SIZE = 20
GENERATIONS = 10
CXPB = 0.5  # Crossover probability
MUTPB = 0.2  # Mutation probability

# Number of folds for Cross-Validation
K_FOLDS = 5

# ================================
# Function Definitions
# ================================

def compute_K_M(betti_sum, num_vertices, dimension, constant_factor, interaction_strength):
    """
    Computes the curvature index K_M based on the sum of Betti numbers, number of vertices, and dimension.
    """
    K_M = constant_factor * (betti_sum + interaction_strength * num_vertices * dimension)
    return K_M

def run_specific_tests(test_cases, constant_factor, interaction_strength, train_indices, valid_indices):
    """
    Runs specific test cases with known homotopical complexities using cross-validation splits.

    Parameters:
    - test_cases: List of dictionaries containing test case information.
    - constant_factor: Current value of CONSTANT_FACTOR.
    - interaction_strength: Current value of INTERACTION_STRENGTH.
    - train_indices: Indices of training test cases.
    - valid_indices: Indices of validation test cases.

    Returns:
    - training_results: Results on training set.
    - validation_results: Results on validation set.
    """
    training_results = []
    validation_results = []

    # Split test_cases into training and validation based on indices
    for idx, case in enumerate(test_cases):
        type_name = case["Type"]
        num_vertices = case["Num_Vertices"]
        dimension = case["Dimension"]
        betti_numbers = case["Betti_Numbers"]
        betti_sum = sum(betti_numbers)
        known_homotopy = case["Known_Homotopy_Groups"]

        # Compute K_M
        K_M = compute_K_M(
            betti_sum=betti_sum,
            num_vertices=num_vertices,
            dimension=dimension,
            constant_factor=constant_factor,
            interaction_strength=interaction_strength
        )

        # Initialize pass condition
        pass_conditions = []

        # Iterate over known homotopy groups and check the invariant
        for n, pi_n in known_homotopy.items():
            # As per Lemma: K_M >= c * pi_n(M) * n
            condition = K_M >= (pi_n * n)
            pass_conditions.append(condition)

        # Overall condition holds if all individual conditions hold
        overall_condition = all(pass_conditions)

        # Append results
        result = {
            "Trial": idx + 1,
            "Type": type_name,
            "Num_Vertices": num_vertices,
            "Dimension": dimension,
            "Betti_Numbers": betti_numbers,
            "Sum_Betti": betti_sum,
            "Known_Homotopy_Groups": known_homotopy,
            "K_M": round(K_M, 2),
            "Bound_Holds": overall_condition
        }

        if idx in train_indices:
            training_results.append(result)
        else:
            validation_results.append(result)

    return training_results, validation_results

def run_comparative_tests(specific_results, edge_case_results, constant_factor, interaction_strength):
    """
    Compares K_invariant with the sum of Betti numbers as a basic invariant.
    """
    results = []

    combined_results = specific_results + edge_case_results

    for result in combined_results:
        K_M = result["K_M"]
        sum_betti = result["Sum_Betti"]
        type_name = result["Type"]

        # Basic invariant: Sum of Betti numbers
        basic_invariant = sum_betti

        # Compare
        comparison = K_M >= basic_invariant

        # Append results
        results.append({
            "Type": type_name,
            "K_M": K_M,
            "Basic_Invariant_Sum_Betti": basic_invariant,
            "K_M >= Sum_Betti": comparison
        })

    return results

def visualize_results(specific_results, edge_case_results, optimized_params):
    """
    Visualizes the relationship between K_M and pi_n(M) * n.
    """
    pi_n_n = []
    K_M_values = []
    labels = []

    # Combine specific and edge case results
    combined_results = specific_results + edge_case_results

    for result in combined_results:
        for n, pi_n in result["Known_Homotopy_Groups"].items():
            pi_n_n.append(pi_n * n)
            # Recompute K_M with optimized parameters
            K_M = compute_K_M(
                betti_sum=result["Sum_Betti"],
                num_vertices=result["Num_Vertices"],
                dimension=result["Dimension"],
                constant_factor=optimized_params['CONSTANT_FACTOR'],
                interaction_strength=optimized_params['INTERACTION_STRENGTH']
            )
            K_M_values.append(K_M)
            labels.append(result["Type"])

    plt.figure(figsize=(10, 6))
    plt.scatter(pi_n_n, K_M_values, color='blue')
    for i, label in enumerate(labels):
        plt.annotate(label, (pi_n_n[i], K_M_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # Plot the line K_M = pi_n(M) * n
    max_val = max(pi_n_n + K_M_values) + 10
    plt.plot([0, max_val], [0, max_val], 'r--', label='K_M = pi_n(M) * n')
    plt.xlabel('pi_n(M) * n')
    plt.ylabel('K_M')
    plt.title('Validation of K_invariant Against Homotopy Groups')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("K_invariant_Validation_Plot.png")
    plt.show()

def compile_and_save_results(specific_results, edge_case_results, comparative_results, optimized_params):
    """
    Compiles all test results and saves them to CSV files.
    """
    # Convert lists of dictionaries to DataFrames
    df_specific = pd.DataFrame(specific_results)
    df_edge_cases = pd.DataFrame(edge_case_results)
    df_comparative = pd.DataFrame(comparative_results)

    # Save to CSV
    df_specific.to_csv("K_invariant_Specific_Test_Results.csv", index=False)
    df_edge_cases.to_csv("K_invariant_Edge_Case_Test_Results.csv", index=False)
    df_comparative.to_csv("K_invariant_Comparative_Test_Results.csv", index=False)

    # Save optimized parameters
    with open("K_invariant_Optimized_Params.txt", "w") as f:
        f.write(f"Optimized CONSTANT_FACTOR: {optimized_params['CONSTANT_FACTOR']}\n")
        f.write(f"Optimized INTERACTION_STRENGTH: {optimized_params['INTERACTION_STRENGTH']}\n")

def expand_test_suite():
    """
    Expands the test suite with additional specific test cases.
    """
    test_cases = [
        # Existing Specific Test Cases
        {
            "Type": "Sphere S^n",
            "Num_Vertices": 20,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 1, 0],  # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 1}  # pi_3(S^3) = Z
        },
        {
            "Type": "Torus T^2",
            "Num_Vertices": 30,
            "Dimension": 2,
            "Betti_Numbers": [1, 2, 1],      # H_0, H_1, H_2
            "Known_Homotopy_Groups": {1: 2}  # pi_1(T^2) = Z^2
        },
        {
            "Type": "Projective Space RP^4",
            "Num_Vertices": 25,
            "Dimension": 4,
            "Betti_Numbers": [1, 0, 0, 1, 0], # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {4: 1}  # pi_4(RP^4) = Z
        },
        {
            "Type": "Wedge Sum of Spheres",
            "Num_Vertices": 35,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 3, 0],    # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 3}  # pi_3(Wedge_S3^3) = Z^3
        },
        {
            "Type": "Complex Torus T^4",
            "Num_Vertices": 40,
            "Dimension": 4,
            "Betti_Numbers": [1, 4, 6, 4, 1],  # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {1: 4}  # pi_1(T^4) = Z^4
        },
        # Additional Specific Test Cases
        {
            "Type": "Eilenberg-MacLane Space K(Z,2)",
            "Num_Vertices": 50,
            "Dimension": 2,
            "Betti_Numbers": [1, 0, 1],        # H_0, H_1, H_2
            "Known_Homotopy_Groups": {2: 1}    # pi_2(K(Z,2)) = Z
        },
        {
            "Type": "Suspension of Torus",
            "Num_Vertices": 45,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 2, 0],     # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 2}    # pi_3(Suspension(T^2)) = 2
        },
        {
            "Type": "Bouquet of Spheres",
            "Num_Vertices": 30,
            "Dimension": 2,
            "Betti_Numbers": [1, 0, 5],        # H_0, H_1, H_2
            "Known_Homotopy_Groups": {2: 5}    # pi_2(Bouquet_S2^5) = Z^5
        },
        {
            "Type": "Configuration Space F(R^2, 3)",
            "Num_Vertices": 60,
            "Dimension": 4,
            "Betti_Numbers": [1, 0, 6, 0, 1],  # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {4: 1}    # pi_4(F(R^2,3)) = Z
        }
    ]

    return test_cases

def introduce_more_edge_cases():
    """
    Introduces more edge cases with high homotopy group values and non-standard spaces.
    """
    edge_cases = [
        # High Homotopy Group Values
        {
            "Type": "High Homotopy Number Space 1",
            "Num_Vertices": 10,
            "Dimension": 2,
            "Betti_Numbers": [1, 10, 1],       # H_0, H_1, H_2
            "Known_Homotopy_Groups": {2: 10}   # pi_2(M) = 10 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 2",
            "Num_Vertices": 15,
            "Dimension": 3,
            "Betti_Numbers": [1, 15, 1, 0],    # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {3: 15}   # pi_3(M) = 15 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 3",
            "Num_Vertices": 20,
            "Dimension": 4,
            "Betti_Numbers": [1, 20, 1, 0, 0], # H_0, H_1, H_2, H_3, H_4
            "Known_Homotopy_Groups": {4: 20}   # pi_4(M) = 20 (hypothetical)
        },
        {
            "Type": "High Homotopy Number Space 4",
            "Num_Vertices": 25,
            "Dimension": 5,
            "Betti_Numbers": [1, 25, 1, 0, 0, 0], # H_0, H_1, H_2, H_3, H_4, H_5
            "Known_Homotopy_Groups": {5: 25}      # pi_5(M) = 25 (hypothetical)
        },
        # Non-Standard Spaces
        {
            "Type": "Exotic Sphere",
            "Num_Vertices": 50,
            "Dimension": 7,
            "Betti_Numbers": [1, 0, 0, 0, 0, 0, 0, 1], # H_0 to H_7
            "Known_Homotopy_Groups": {7: 1}          # pi_7(Exotic_Sphere) = Z
        },
        {
            "Type": "Lens Space L(p, q)",
            "Num_Vertices": 35,
            "Dimension": 3,
            "Betti_Numbers": [1, 0, 1, 0],            # H_0, H_1, H_2, H_3
            "Known_Homotopy_Groups": {1: 1}           # pi_1(L(p,q)) = Z/pZ
        }
    ]

    return edge_cases

def parameter_optimization_optuna(specific_results, edge_case_results):
    """
    Optimizes CONSTANT_FACTOR and INTERACTION_STRENGTH using Bayesian Optimization (Optuna) with Cross-Validation.
    """
    def objective(trial):
        c_factor = trial.suggest_float('CONSTANT_FACTOR', 0.5, 3.0, step=0.1)
        i_strength = trial.suggest_float('INTERACTION_STRENGTH', 0.1, 1.0, step=0.1)

        combined_results = specific_results + edge_case_results
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        score = 0

        # Convert combined_results to a list
        results_list = combined_results.copy()

        for train_index, valid_index in kf.split(results_list):
            training_results, validation_results = run_specific_tests(
                test_cases=results_list,
                constant_factor=c_factor,
                interaction_strength=i_strength,
                train_indices=train_index,
                valid_indices=valid_index
            )
            # Evaluate on validation set
            for result in validation_results:
                score += int(result["Bound_Holds"])

        return score

    study = optuna.create_study(direction='maximize', sampler=OPTUNA_SAMPLER)
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value

    print("\n=== Parameter Optimization (Optuna) Completed ===")
    print(f"Best Score: {best_score} out of {len(specific_results) + len(edge_case_results)}")
    print(f"Best Parameters: CONSTANT_FACTOR = {best_params['CONSTANT_FACTOR']}, INTERACTION_STRENGTH = {best_params['INTERACTION_STRENGTH']}")

    return best_params

def parameter_optimization_genetic(specific_results, edge_case_results):
    """
    Optimizes CONSTANT_FACTOR and INTERACTION_STRENGTH using Genetic Algorithms (DEAP) with Cross-Validation.
    """
    # Define evaluation function
    def evaluate(individual):
        c_factor, i_strength = individual
        score = 0
        combined_results = specific_results + edge_case_results
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        results_list = combined_results.copy()

        for train_index, valid_index in kf.split(results_list):
            training_results, validation_results = run_specific_tests(
                test_cases=results_list,
                constant_factor=c_factor,
                interaction_strength=i_strength,
                train_indices=train_index,
                valid_indices=valid_index
            )
            for result in validation_results:
                score += int(result["Bound_Holds"])

        return (score,)

    # Setup DEAP framework
    # Check if creator already has FitnessMax_GA and Individual_GA to avoid duplication
    try:
        creator.create("FitnessMax_GA", base.Fitness, weights=(1.0,))
        creator.create("Individual_GA", list, fitness=creator.FitnessMax_GA)
    except AttributeError:
        pass  # FitnessMax_GA and Individual_GA already exist

    toolbox = base.Toolbox()

    # Attribute generators
    toolbox.register("attr_c_factor", random.uniform, 0.5, 3.0)
    toolbox.register("attr_i_strength", random.uniform, 0.1, 1.0)

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual_GA,
                     (toolbox.attr_c_factor, toolbox.attr_i_strength), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation, crossover, mutation, selection
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)

    # Define statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Run Genetic Algorithm
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                              ngen=GENERATIONS, stats=stats, verbose=False)

    # Extract best individual
    top_ind = tools.selBest(population, k=1)[0]
    best_score = top_ind.fitness.values[0]
    best_params = {
        'CONSTANT_FACTOR': round(top_ind[0], 2),
        'INTERACTION_STRENGTH': round(top_ind[1], 2)
    }

    print("\n=== Parameter Optimization (Genetic Algorithm) Completed ===")
    print(f"Best Score: {best_score} out of {len(specific_results) + len(edge_case_results)}")
    print(f"Best Parameters: CONSTANT_FACTOR = {best_params['CONSTANT_FACTOR']}, INTERACTION_STRENGTH = {best_params['INTERACTION_STRENGTH']}")

    return best_params

def parameter_optimization(specific_results, edge_case_results):
    """
    Optimizes CONSTANT_FACTOR and INTERACTION_STRENGTH using both Optuna (Bayesian) and Genetic Algorithms with Cross-Validation.
    Chooses the best parameters from both methods.
    """
    # Bayesian Optimization with Optuna
    optuna_params = parameter_optimization_optuna(specific_results, edge_case_results)
    optuna_score = sum([
        1 for result in specific_results + edge_case_results
        if compute_K_M(
            betti_sum=result["Sum_Betti"],
            num_vertices=result["Num_Vertices"],
            dimension=result["Dimension"],
            constant_factor=optuna_params['CONSTANT_FACTOR'],
            interaction_strength=optuna_params['INTERACTION_STRENGTH']
        ) >= max([pi_n * n for n, pi_n in result["Known_Homotopy_Groups"].items()])
    ])

    # Genetic Algorithm Optimization with DEAP
    genetic_params = parameter_optimization_genetic(specific_results, edge_case_results)
    genetic_score = sum([
        1 for result in specific_results + edge_case_results
        if compute_K_M(
            betti_sum=result["Sum_Betti"],
            num_vertices=result["Num_Vertices"],
            dimension=result["Dimension"],
            constant_factor=genetic_params['CONSTANT_FACTOR'],
            interaction_strength=genetic_params['INTERACTION_STRENGTH']
        ) >= max([pi_n * n for n, pi_n in result["Known_Homotopy_Groups"].items()])
    ])

    # Select the better optimization result
    if optuna_score >= genetic_score:
        best_params = optuna_params
        best_score = optuna_score
        method = "Optuna (Bayesian Optimization)"
    else:
        best_params = genetic_params
        best_score = genetic_score
        method = "Genetic Algorithm"

    print(f"\n=== Parameter Optimization Result ===")
    print(f"Best Parameters from {method}: CONSTANT_FACTOR = {best_params['CONSTANT_FACTOR']}, INTERACTION_STRENGTH = {best_params['INTERACTION_STRENGTH']}")
    print(f"Score: {best_score} out of {len(specific_results) + len(edge_case_results)}")

    return best_params

# ================================
# Main Execution Function
# ================================

def main():
    """
    Main function to execute all testing, optimization, comparison, visualization, and reporting.
    """
    # Step 1: Expand Test Suite with Additional Specific Test Cases
    test_cases = expand_test_suite()

    # Step 2: Introduce More Edge Cases
    edge_case_tests = introduce_more_edge_cases()

    # Step 3: Run Specific Tests with Initial Parameters
    print("=== Running Specific Tests ===")
    training_results, validation_results = run_specific_tests(
        test_cases=test_cases[:5],  # First 5 are original specific tests
        constant_factor=INITIAL_CONSTANT_FACTOR,
        interaction_strength=INITIAL_INTERACTION_STRENGTH,
        train_indices=list(range(len(test_cases[:5]))),  # All as training in initial run
        valid_indices=[]  # No validation initially
    )

    # Combine training and validation results
    specific_results_initial = training_results + validation_results

    # Additional Specific Test Cases
    additional_training_results, additional_validation_results = run_specific_tests(
        test_cases=test_cases[5:],  # Remaining are additional specific tests
        constant_factor=INITIAL_CONSTANT_FACTOR,
        interaction_strength=INITIAL_INTERACTION_STRENGTH,
        train_indices=list(range(len(test_cases[5:]))),  # All as training in initial run
        valid_indices=[]  # No validation initially
    )

    # Combine additional training and validation results
    additional_specific_results_initial = additional_training_results + additional_validation_results

    # Extend the specific_results_initial with additional results
    specific_results_initial.extend(additional_specific_results_initial)

    # Step 4: Run Edge Case Tests with Initial Parameters
    print("=== Running Edge Case Tests ===")
    edge_case_training_results, edge_case_validation_results = run_specific_tests(
        test_cases=edge_case_tests,
        constant_factor=INITIAL_CONSTANT_FACTOR,
        interaction_strength=INITIAL_INTERACTION_STRENGTH,
        train_indices=list(range(len(edge_case_tests))),  # All as training in initial run
        valid_indices=[]  # No validation initially
    )

    # Combine edge case training and validation results
    edge_case_results_initial = edge_case_training_results + edge_case_validation_results

    # Step 5: Parameter Optimization
    print("=== Optimizing Parameters ===")
    optimized_params = parameter_optimization(specific_results_initial, edge_case_results_initial)

    # Step 6: Re-run Specific and Edge Case Tests with Optimized Parameters
    print("=== Re-running Specific Tests with Optimized Parameters ===")
    training_results_opt, validation_results_opt = run_specific_tests(
        test_cases=test_cases[:5],
        constant_factor=optimized_params['CONSTANT_FACTOR'],
        interaction_strength=optimized_params['INTERACTION_STRENGTH'],
        train_indices=list(range(len(test_cases[:5]))),
        valid_indices=[]
    )

    # Combine training and validation results
    specific_results_optimized = training_results_opt + validation_results_opt

    # Additional Specific Test Cases
    additional_training_opt, additional_validation_opt = run_specific_tests(
        test_cases=test_cases[5:],  # Remaining are additional specific tests
        constant_factor=optimized_params['CONSTANT_FACTOR'],
        interaction_strength=optimized_params['INTERACTION_STRENGTH'],
        train_indices=list(range(len(test_cases[5:]))),
        valid_indices=[]
    )

    # Combine additional training and validation results
    additional_specific_results_optimized = additional_training_opt + additional_validation_opt

    # Extend the specific_results_optimized with additional results
    specific_results_optimized.extend(additional_specific_results_optimized)

    print("=== Re-running Edge Case Tests with Optimized Parameters ===")
    edge_case_training_opt, edge_case_validation_opt = run_specific_tests(
        test_cases=edge_case_tests,
        constant_factor=optimized_params['CONSTANT_FACTOR'],
        interaction_strength=optimized_params['INTERACTION_STRENGTH'],
        train_indices=list(range(len(edge_case_tests))),
        valid_indices=[]
    )

    # Combine edge case training and validation results
    edge_case_results_optimized = edge_case_training_opt + edge_case_validation_opt

    # Step 7: Comparative Analysis
    print("=== Running Comparative Analysis ===")
    comparative_results = run_comparative_tests(
        specific_results=specific_results_optimized,
        edge_case_results=edge_case_results_optimized,
        constant_factor=optimized_params['CONSTANT_FACTOR'],
        interaction_strength=optimized_params['INTERACTION_STRENGTH']
    )

    # Step 8: Visualization
    print("=== Generating Visualization ===")
    visualize_results(specific_results_optimized, edge_case_results_optimized, optimized_params)

    # Step 9: Compile and Save Results
    print("=== Compiling and Saving Results ===")
    compile_and_save_results(specific_results_optimized, edge_case_results_optimized, comparative_results, optimized_params)

    # Step 10: Print Summary
    total_specific = len(specific_results_optimized)
    passed_specific = sum([res["Bound_Holds"] for res in specific_results_optimized])
    total_edge = len(edge_case_results_optimized)
    passed_edge = sum([res["Bound_Holds"] for res in edge_case_results_optimized])
    total_comparative = len(comparative_results)
    passed_comparative = sum([res["K_M >= Sum_Betti"] for res in comparative_results])

    print("\n=== Comprehensive Test Summary ===")
    print(f"Specific Tests: {passed_specific}/{total_specific} Passed")
    print(f"Edge Case Tests: {passed_edge}/{total_edge} Passed")
    print(f"Comparative Tests (K_M >= Sum_Betti): {passed_comparative}/{total_comparative} Passed")

    print("\nAll tests completed successfully. Results have been saved to CSV files and a visualization plot has been generated.")

# ================================
# Execute the Script
# ================================

if __name__ == "__main__":
    main()
