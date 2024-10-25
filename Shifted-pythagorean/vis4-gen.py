import numpy as np
from numba import cuda, int32, float64
import math
from math import sqrt
from collections import defaultdict
import time

# Constants
MAX_Z = 1_000_000_000
MIN_FAMILY_SIZE = 23
THREADS_PER_BLOCK = 512

# CUDA Kernel to count solutions and compute y_max and y_min for each z
@cuda.jit
def find_solutions_kernel(z_values, solution_counts, y_max_values, y_min_values):
    z_idx = cuda.grid(1)
    if z_idx < z_values.size:
        z = z_values[z_idx]
        count = 0
        y_max = 0
        y_min = z  # Initialize to z, as y cannot be greater than z in this context

        # Compute upper bound for x using sqrt and casting to int
        upper_x = int(math.sqrt(z * z + 1))

        for x in range(2, upper_x):  # Start x at 2 to exclude y = z
            y_squared = z * z + 1 - x * x
            if y_squared > 0:
                y = int(math.sqrt(y_squared))
                if y * y == y_squared and y > x:
                    count += 1
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

        solution_counts[z_idx] = count
        y_max_values[z_idx] = y_max
        y_min_values[z_idx] = y_min

def main():
    # Generate all z values as float64 to prevent overflow
    z_cpu = np.arange(1, MAX_Z + 1, dtype=np.float64)

    # Allocate device memory using device_array with appropriate dtype
    solution_counts_gpu = cuda.device_array(z_cpu.shape, dtype=np.int32)
    y_max_gpu = cuda.device_array(z_cpu.shape, dtype=np.int32)
    y_min_gpu = cuda.device_array(z_cpu.shape, dtype=np.int32)

    # Define grid size
    blocks_per_grid = (MAX_Z + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"Launching CUDA kernel with {blocks_per_grid} blocks and {THREADS_PER_BLOCK} threads per block...")
    start_time = time.time()

    # Launch the kernel
    find_solutions_kernel[blocks_per_grid, THREADS_PER_BLOCK](z_cpu, solution_counts_gpu, y_max_gpu, y_min_gpu)

    # Wait for the kernel to complete
    cuda.synchronize()
    kernel_time = time.time() - start_time
    print(f"CUDA kernel completed in {kernel_time:.2f} seconds.")

    # Retrieve results from the GPU
    solution_counts = solution_counts_gpu.copy_to_host()
    y_max_values = y_max_gpu.copy_to_host()
    y_min_values = y_min_gpu.copy_to_host()

    # Filter z values with large families (size >= MIN_FAMILY_SIZE)
    large_families_indices = np.where(solution_counts >= MIN_FAMILY_SIZE)[0]
    large_families_z = z_cpu[large_families_indices]
    large_families_counts = solution_counts[large_families_indices]
    large_families_y_max = y_max_values[large_families_indices]
    large_families_y_min = y_min_values[large_families_indices]

    # Compute y_max/y_min ratio and error from sqrt(2)
    y_ratio = large_families_y_max / large_families_y_min
    error_from_sqrt2 = np.abs(y_ratio - sqrt(2))

    # Aggregate error_from_sqrt2 into a NumPy array for statistical analysis
    error_array = error_from_sqrt2

    # Calculate statistical summary
    summary = {
        'Total Families': len(error_array),
        'Mean Error from √2': np.mean(error_array),
        'Median Error from √2': np.median(error_array),
        'Standard Deviation of Error': np.std(error_array),
        'Minimum Error': np.min(error_array),
        'Maximum Error': np.max(error_array)
    }

    # Print the statistical summary
    print("\n--- Statistical Summary of Error from √2 ---")
    for key, value in summary.items():
        print(f"{key}: {value:.10f}")

    # Optionally, provide sample entries for verification
    print("\nSample Entries for Each Family Size:")
    family_summary = defaultdict(list)
    for idx, family_size in enumerate(large_families_counts):
        family_summary[family_size].append({
            'z': int(large_families_z[idx]),
            'y_max/y_min': y_ratio[idx],
            'error_from_sqrt2': error_from_sqrt2[idx]
        })

    # Define how many samples you want per family size
    SAMPLES_PER_FAMILY = 10

    for family_size in sorted(family_summary.keys(), reverse=True):
        print(f"Family Size: {family_size}")
        samples = family_summary[family_size][:SAMPLES_PER_FAMILY]
        for sample in samples:
            print(f"  z: {sample['z']}, y_max/y_min: {sample['y_max/y_min']:.5f}, error from sqrt(2): {sample['error_from_sqrt2']:.10f}")
        print()

    # Analyze family size distribution
    print("Analyzing family size distribution...")
    family_size_distribution = analyze_family_sizes(solution_counts)
    print(f"Family size distribution: {family_size_distribution}")

    total_time = time.time() - start_time
    print(f"Total analysis complete in {total_time:.2f} seconds.")

def analyze_family_sizes(solution_counts, min_family_size=MIN_FAMILY_SIZE):
    family_sizes = defaultdict(int)
    for count in solution_counts:
        if count >= min_family_size:
            family_sizes[count] += 1
    return dict(family_sizes)

if __name__ == "__main__":
    main()
