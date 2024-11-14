import numpy as np
import gudhi as gd
import pandas as pd
from tqdm import tqdm

def calculate_betti_numbers_with_gudhi(num_vertices, dimension):
    points = np.random.rand(num_vertices, dimension)
    max_edge_length = 0.5
    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)
    simplex_tree.compute_persistence()
    betti_numbers_list = simplex_tree.betti_numbers()
    # Ensure betti_numbers_list has entries up to 'dimension'
    betti_numbers = betti_numbers_list + [0]*(dimension - len(betti_numbers_list))
    betti_numbers = betti_numbers[:dimension]
    return betti_numbers

def simulate_complexes_with_gudhi(num_tests=1000, constant_factor=2.5, interaction_strength=0.7):
    results = []
    for _ in tqdm(range(num_tests)):
        num_vertices = np.random.randint(10, 30)
        dimension = np.random.randint(2, 6)
        betti_numbers = calculate_betti_numbers_with_gudhi(num_vertices, dimension)
        complexity = sum(betti_numbers)
        K_M = constant_factor * (complexity + interaction_strength * num_vertices * np.exp(dimension))
        bound_check = K_M >= complexity
        results.append({
            "num_vertices": num_vertices,
            "dimension": dimension,
            "betti_numbers": betti_numbers,
            "complexity": complexity,
            "K_M": K_M,
            "bound_check": bound_check
        })
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    df = simulate_complexes_with_gudhi(num_tests=1000, constant_factor=2.5, interaction_strength=0.7)
    success_rate = df["bound_check"].mean() * 100
    print(f"Success Rate: {success_rate:.2f}%")
    print(df.head())
