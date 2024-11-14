import numpy as np
from scipy.linalg import expm

def construct_chevalley_basis(group_name):
    if group_name == 'G2':
        # G2 Chevalley basis, simplified 7x7 matrices
        generators = []

        h1 = np.diag([1, 1, 0, 0, 0, 0, 0])
        h2 = np.diag([0, 0, 1, 1, 0, 0, 0])

        e1 = np.zeros((7, 7))
        e1[0, 1] = 1
        e1[1, 2] = 1
        e1[2, 3] = 1

        e2 = np.zeros((7, 7))
        e2[3, 4] = 1
        e2[4, 5] = 1
        e2[5, 6] = 1

        generators.extend([h1, h2, e1, e2])
        return generators

    elif group_name == 'F4':
        # F4 Chevalley basis, simplified 26x26 matrices
        generators = []

        h1 = np.diag([1] * 13 + [0] * 13)
        h2 = np.diag([0] * 13 + [1] * 13)

        e1 = np.zeros((26, 26))
        e1[0, 1] = 1
        e1[1, 2] = 1
        e1[2, 3] = 1
        e1[3, 4] = 1

        e2 = np.zeros((26, 26))
        e2[13, 14] = 1
        e2[14, 15] = 1
        e2[15, 16] = 1
        e2[16, 17] = 1

        generators.extend([h1, h2, e1, e2])
        return generators

    elif group_name == 'E6':
        # E6 Chevalley basis, simplified 27x27 matrices
        generators = []

        h1 = np.diag([1] * 9 + [0] * 18)
        h2 = np.diag([0] * 9 + [1] * 18)

        e1 = np.zeros((27, 27))
        e1[0, 1] = 1
        e1[1, 2] = 1
        e1[2, 3] = 1

        e2 = np.zeros((27, 27))
        e2[9, 10] = 1
        e2[10, 11] = 1
        e2[11, 12] = 1

        generators.extend([h1, h2, e1, e2])
        return generators

    elif group_name == 'E7':
        # E7 Chevalley basis, simplified 56x56 matrices
        generators = []

        h1 = np.diag([1] * 28 + [0] * 28)
        h2 = np.diag([0] * 28 + [1] * 28)

        e1 = np.zeros((56, 56))
        e1[0, 1] = 1
        e1[1, 2] = 1
        e1[2, 3] = 1

        e2 = np.zeros((56, 56))
        e2[28, 29] = 1
        e2[29, 30] = 1
        e2[30, 31] = 1

        generators.extend([h1, h2, e1, e2])
        return generators

    elif group_name == 'E8':
        # E8 Chevalley basis, simplified 248x248 matrices
        generators = []

        h1 = np.diag([1] * 124 + [0] * 124)
        h2 = np.diag([0] * 124 + [1] * 124)

        e1 = np.zeros((248, 248))
        e1[0, 1] = 1
        e1[1, 2] = 1
        e1[2, 3] = 1

        e2 = np.zeros((248, 248))
        e2[124, 125] = 1
        e2[125, 126] = 1
        e2[126, 127] = 1

        generators.extend([h1, h2, e1, e2])
        return generators

    else:
        raise ValueError(f"Unsupported group: {group_name}")

# Generate Lie group elements via exponentiating Lie algebra combinations
def generate_group_elements(basis_generators, num_samples):
    samples = []
    for _ in range(num_samples):
        linear_combo = sum(np.random.rand() * gen for gen in basis_generators)
        group_element = expm(linear_combo)
        samples.append(group_element)
    return samples

# Compute conjugacy classes for each generated group element
def generate_conjugacy_classes(samples):
    conjugacy_classes = []
    for sample in samples:
        class_elements = [sample]
        for _ in range(10):
            random_transform = np.random.rand(*sample.shape)
            conj = random_transform @ sample @ np.linalg.inv(random_transform)
            class_elements.append(conj)
        conjugacy_classes.extend(class_elements)
    return conjugacy_classes

# Calculate entropy using Shannon entropy on eigenvalue phases
def calculate_entropy(conjugacy_classes):
    eigenvalue_phases = []
    for matrix in conjugacy_classes:
        eigenvalues = np.linalg.eigvals(matrix)
        phases = np.angle(eigenvalues)
        eigenvalue_phases.extend(phases)
    eigenvalue_phases = np.array(eigenvalue_phases).reshape(-1, 1)

    hist, bin_edges = np.histogram(eigenvalue_phases, bins=100, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist) * (bin_edges[1] - bin_edges[0]))

    return entropy

# Main function to run calculations for various exceptional groups
def main():
    num_samples = 50  # Adjust as needed

    groups = ['G2', 'F4', 'E6', 'E7', 'E8']  # Exceptional groups

    for group_name in groups:
        print(f"Calculating SOE for {group_name}...")

        # Step 1: Construct Lie algebra basis
        basis_generators = construct_chevalley_basis(group_name)

        # Step 2: Generate group elements
        samples = generate_group_elements(basis_generators, num_samples)

        # Step 3: Generate conjugacy classes
        conjugacy_classes = generate_conjugacy_classes(samples)

        # Step 4: Calculate entropy
        entropy = calculate_entropy(conjugacy_classes)

        print(f"Symmetry Orbit Entropy for {group_name}: {entropy}")

if __name__ == "__main__":
    main()
