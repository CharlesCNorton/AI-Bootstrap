import numpy as np
import cupy as cp
from math import sqrt, isqrt
from typing import List, Dict, Tuple
from collections import defaultdict
import time

# Configure CuPy to use most of available GPU memory
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=128*1024*1024*1024)  # 45GB of VRAM

def find_solutions_batch_cuda(start_z: int, end_z: int) -> Dict[int, List[Tuple[int, int]]]:
    """Find solutions for a range of z values using CUDA"""
    z_values = cp.arange(start_z, end_z, dtype=cp.int64)
    z_squared_plus_1 = z_values * z_values + 1

    solutions_dict = {}
    batch_size = min(1000, end_z - start_z)

    for batch_start in range(0, len(z_values), batch_size):
        batch_end = min(batch_start + batch_size, len(z_values))
        batch_z = z_values[batch_start:batch_end]
        batch_z_squared_plus_1 = z_squared_plus_1[batch_start:batch_end]

        # Maximum possible x for this batch
        max_x = int(cp.max(batch_z).get())
        x_array = cp.arange(2, max_x, dtype=cp.int64)

        # Calculate y_squared for all combinations
        y_squared = cp.expand_dims(batch_z_squared_plus_1, 1) - cp.expand_dims(x_array * x_array, 0)

        # Find valid y values
        valid_y = cp.sqrt(y_squared)
        valid_y_int = cp.floor(valid_y).astype(cp.int64)

        # Check perfect squares
        is_perfect_square = (valid_y_int * valid_y_int == y_squared) & (y_squared > 0)

        # Process each z in the batch
        for idx, z in enumerate(batch_z.get()):
            z_solutions = []
            x_indices, = cp.where(is_perfect_square[idx])

            if len(x_indices) > 0:
                x_values = x_array[x_indices].get()
                y_values = valid_y_int[idx, x_indices].get()

                for x, y in zip(x_values, y_values):
                    if y > x:
                        z_solutions.append((int(x), int(y)))

            if len(z_solutions) >= 20:  # Only store families with 20+ solutions
                solutions_dict[int(z)] = sorted(z_solutions)

        # Free memory explicitly
        mempool.free_all_blocks()

    return solutions_dict

def analyze_family_sizes_cuda(max_z: int = 1_000_000, min_solutions: int = 20) -> Dict:
    """Analyze family sizes using CUDA acceleration"""
    print(f"Analyzing families up to z={max_z:,}...")

    # Process in chunks to manage memory
    chunk_size = 100_000
    families = defaultdict(list)
    sqrt2_patterns = defaultdict(list)

    for chunk_start in range(2, max_z + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, max_z + 1)

        # Find solutions for this chunk
        solutions_dict = find_solutions_batch_cuda(chunk_start, chunk_end)

        # Analyze solutions
        for z, solutions in solutions_dict.items():
            size = len(solutions)
            if size >= min_solutions:
                families[size].append(z)

                # Analyze √2 relationship
                y_values = [y for _, y in solutions]
                actual_ratio = max(y_values) / min(y_values)
                sqrt2 = sqrt(2)

                # Find closest √2 power
                ratios = []
                for power in range(-4, 5):
                    target = sqrt2 ** power
                    error = abs(actual_ratio - target)
                    ratios.append((power, target, error))

                best_fit = min(ratios, key=lambda x: x[2])

                sqrt2_patterns[size].append({
                    'actual_ratio': actual_ratio,
                    'closest_sqrt2_power': best_fit[0],
                    'closest_sqrt2_value': best_fit[1],
                    'error': best_fit[2]
                })

        print(f"Progress: {chunk_end/max_z*100:.1f}%")
        mempool.free_all_blocks()

    # Analyze patterns by family size
    results = {}
    for size, z_values in families.items():
        patterns = sqrt2_patterns[size]
        ratios = [p['actual_ratio'] for p in patterns]
        errors = [p['error'] for p in patterns]

        results[size] = {
            'count': len(z_values),
            'z_values': z_values,
            'ratio_stats': {
                'mean': float(np.mean(ratios)),
                'std': float(np.std(ratios)),
                'min': float(min(ratios)),
                'max': float(max(ratios))
            },
            'error_stats': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(min(errors)),
                'max': float(max(errors))
            }
        }

    return results

def main():
    start_time = time.time()

    # Run family size analysis
    results = analyze_family_sizes_cuda(1_000_000, 20)

    print("\n=== FAMILY SIZE ANALYSIS ===")
    print("\nSize Distribution and Ratio Patterns:")
    for size in sorted(results.keys(), reverse=True):
        data = results[size]
        print(f"\nSize {size}:")
        print(f"Count: {data['count']:,}")
        print(f"Ratio mean: {data['ratio_stats']['mean']:.8f} ± {data['ratio_stats']['std']:.8f}")
        print(f"Error mean: {data['error_stats']['mean']:.8f} ± {data['error_stats']['std']:.8f}")
        print(f"First few z-values: {data['z_values'][:5]}")

    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
