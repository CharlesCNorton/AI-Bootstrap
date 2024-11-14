import snappy
import numpy as np
import gudhi
import itertools
import logging
from tqdm import tqdm
import multiprocessing
import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Generate a List of Knot Complements Using SnapPy
def generate_knot_complements(knot_names):
    manifolds = []
    for knot in knot_names:
        try:
            manifolds.append(snappy.Manifold(knot))
        except RuntimeError as e:
            logging.error(f"Error creating manifold for knot {knot}: {e}")
    return manifolds

# Step 2: Compute Your Invariant for the Knot Complement (as a 3-manifold)
def compute_my_invariant(manifold_name):
    try:
        manifold = snappy.Manifold(manifold_name)
        simplex_tree = gudhi.SimplexTree()

        # Extract triangulation information from SnapPy
        triangulation = manifold.triangulation_isosig()

        # Creating a simplicial complex with dummy data
        num_vertices = len(triangulation)

        for i in range(num_vertices):
            simplex_tree.insert([i])

        simplex_tree.compute_persistence()

        # Compute the persistence features
        persistence_pairs = simplex_tree.persistence()
        betti_numbers = simplex_tree.betti_numbers()
        total_lifetime = sum(death - birth if isinstance(death, (int, float)) else 0
                             for birth, death in persistence_pairs if death != float('inf'))

        # Calculate invariant (dummy version)
        invariant = sum(betti_numbers) + total_lifetime
        return manifold_name, invariant

    except Exception as e:
        logging.error(f"Error processing manifold {manifold_name}: {e}")
        return manifold_name, None

# Step 3: Multiprocessing Worker for Isolating Calculations
def process_knot_complement(manifold_name):
    return compute_my_invariant(manifold_name)

# Step 4: Compare Knot Complements Using Your Invariant
def classify_complements(manifolds):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_knot_complement, [manifold.name() for manifold in manifolds])
    pool.close()
    pool.join()

    invariant_data = {name: inv for name, inv in results if inv is not None}

    comparisons = []
    for (m1, m2) in itertools.combinations(manifolds, 2):
        if m1.name() in invariant_data and m2.name() in invariant_data:
            inv_1 = invariant_data[m1.name()]
            inv_2 = invariant_data[m2.name()]

            # Are the invariants close enough to suggest homeomorphism?
            if abs(inv_1 - inv_2) < 0.01:  # Assuming some error threshold
                comparisons.append((m1.name(), m2.name(), 'Equivalent'))
            else:
                comparisons.append((m1.name(), m2.name(), 'Distinct'))

    return comparisons, invariant_data

# Step 5: Perform Failure Analysis
def failure_analysis(results):
    successful_classifications = 0
    false_positives = 0
    false_negatives = 0

    for result in results:
        knot_1, knot_2, relation = result
        if relation == 'Equivalent':
            successful_classifications += 1
        elif relation == 'Distinct':
            if knot_1 == knot_2:
                false_positives += 1
            else:
                false_negatives += 1

    total_tests = len(results)
    logging.info("=== Knot Complement Comparison Results ===")
    logging.info(f"Total Tests: {total_tests}")
    logging.info(f"Successful Classifications: {successful_classifications}")
    logging.info(f"False Positives: {false_positives}")
    logging.info(f"False Negatives: {false_negatives}")

    if total_tests > 0:
        accuracy = (successful_classifications / total_tests) * 100
        logging.info(f"Accuracy: {accuracy:.2f}%")
    else:
        logging.info("No valid tests were performed.")

# Step 6: Clustering Analysis Summary
def clustering_analysis(invariant_data):
    names = list(invariant_data.keys())
    invariants = list(invariant_data.values())

    # Creating a DataFrame for simplicity
    df = pd.DataFrame({'Name': names, 'Invariant': invariants})

    # Clustering
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(df[['Invariant']])

    # Display clustering results
    for cluster in range(3):
        cluster_data = df[df['Cluster'] == cluster]
        logging.info(f"Cluster {cluster} ({len(cluster_data)} elements):")
        logging.info(f"Average Invariant: {cluster_data['Invariant'].mean():.2f}")
        logging.info(f"Knots in Cluster: {cluster_data['Name'].tolist()}")

    # Output numerical summary for easier analysis
    logging.info("=== Numerical Summary of Clustering ===")
    cluster_summary = df.groupby('Cluster')['Invariant'].agg(['mean', 'std', 'min', 'max', 'count'])
    logging.info(f"\n{cluster_summary}")

# Main Execution
if __name__ == "__main__":
    # Example set of prime knots; you can expand this list
    knot_names = ['3_1', '4_1', '5_2', '6_3', '7_1', '8_19', '9_42']
    manifolds = generate_knot_complements(knot_names)

    # Compare Knot Complements Using Your Invariant
    classification_results, invariant_data = classify_complements(manifolds)

    # Perform Failure Analysis
    failure_analysis(classification_results)

    # Perform Clustering Analysis
    clustering_analysis(invariant_data)
