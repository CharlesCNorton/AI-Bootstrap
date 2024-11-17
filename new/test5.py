import numpy as np
import csv
from ripser import ripser
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# Load the derivatives data from CSV file in smaller parts
subset_size = 5
all_data = []

# Load data in smaller chunks to manage memory
with open('zeta_derivatives.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        all_data.append([float(value) for value in row])

        # Process in subsets to manage memory
        if len(all_data) == subset_size:
            V_rho = np.array(all_data)[:, 1:]  # Exclude 'zero' column, we need derivatives only
            rho_values = np.array(all_data)[:, 0]

            # Normalize V_rho by log(rho) for enhanced analysis
            positive_indices = rho_values > 0
            V_rho_filtered = V_rho[positive_indices, :]
            rho_filtered = rho_values[positive_indices]
            log_rho_filtered = np.log(rho_filtered)

            # Normalize the vector space
            V_rho_enhanced = np.array([V_rho_filtered[:, i] / log_rho_filtered for i in range(V_rho_filtered.shape[1])]).T

            # Run persistent homology on the subset
            result = ripser(V_rho_enhanced)

            # Extract the persistence diagrams
            dgms = result['dgms']

            # Statistical and numerical analysis of the persistence diagrams for each subset
            for i, dgm in enumerate(dgms):
                if len(dgm) == 0:
                    print(f"Dimension {i}: No features detected in this subset.")
                    continue

                print(f"\nSubset processed, Dimension {i}:")
                births = dgm[:, 0]
                deaths = dgm[:, 1]
                lifetimes = deaths - births

                # Numerical details
                print(f"Number of features: {len(dgm)}")
                print(f"Average birth time: {np.mean(births):.5f}")
                print(f"Average death time: {np.mean(deaths):.5f}")
                print(f"Average lifetime: {np.mean(lifetimes):.5f}")
                print(f"Max lifetime: {np.max(lifetimes):.5f}")

                # Persistent pairs details
                print(f"Top birth-death pairs:")
                for b, d in dgm:
                    print(f"Birth: {b:.5f}, Death: {d:.5f}, Lifetime: {d - b:.5f}")

            # Clear subset to process the next batch
            all_data = []

# If there are remaining rows to process after the last full batch
if len(all_data) > 0:
    V_rho = np.array(all_data)[:, 1:]  # Process the remaining data
    rho_values = np.array(all_data)[:, 0]
    positive_indices = rho_values > 0
    V_rho_filtered = V_rho[positive_indices, :]
    rho_filtered = rho_values[positive_indices]
    log_rho_filtered = np.log(rho_filtered)
    V_rho_enhanced = np.array([V_rho_filtered[:, i] / log_rho_filtered for i in range(V_rho_filtered.shape[1])]).T
    result = ripser(V_rho_enhanced)
    dgms = result['dgms']

    # Output the results for the final subset
    for i, dgm in enumerate(dgms):
        if len(dgm) == 0:
            print(f"Dimension {i}: No features detected in the final subset.")
            continue

        print(f"\nFinal subset, Dimension {i}:")
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        lifetimes = deaths - births

        print(f"Number of features: {len(dgm)}")
        print(f"Average birth time: {np.mean(births):.5f}")
        print(f"Average death time: {np.mean(deaths):.5f}")
        print(f"Average lifetime: {np.mean(lifetimes):.5f}")
        print(f"Max lifetime: {np.max(lifetimes):.5f}")
