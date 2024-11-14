import numpy as np
from scipy.linalg import expm
import itertools


# --- Step 1: Define the Cartan Matrix for Exceptional Lie Groups ---

def construct_cartan_matrix(group_name):
    if group_name == 'E8':
        # E8 Cartan matrix (8 x 8), representing the 8 simple roots and their relationships
        return np.array([[2, -1, 0, 0, 0, 0, 0, 0],
                         [-1, 2, -1, 0, 0, 0, 0, 0],
                         [0, -1, 2, -1, 0, 0, 0, 0],
                         [0, 0, -1, 2, -1, 0, 0, 0],
                         [0, 0, 0, -1, 2, -1, 0, 0],
                         [0, 0, 0, 0, -1, 2, -1, 0],
                         [0, 0, 0, 0, 0, -1, 2, -1],
                         [0, 0, 0, 0, 0, 0, -1, 2]])
    elif group_name == 'F4':
        # F4 Cartan matrix (4 x 4)
        return np.array([[2, -1, 0, 0],
                         [-1, 2, -1, 0],
                         [0, -1, 2, -1],
                         [0, 0, -1, 2]])
    elif group_name == 'E6':
        # E6 Cartan matrix (6 x 6)
        return np.array([[2, -1, 0, 0, 0, 0],
                         [-1, 2, -1, 0, 0, 0],
                         [0, -1, 2, -1, 0, 0],
                         [0, 0, -1, 2, -1, 0],
                         [0, 0, 0, -1, 2, -1],
                         [0, 0, 0, 0, -1, 2]])
    elif group_name == 'E7':
        # E7 Cartan matrix (7 x 7)
        return np.array([[2, -1, 0, 0, 0, 0, 0],
                         [-1, 2, -1, 0, 0, 0, 0],
                         [0, -1, 2, -1, 0, 0, 0],
                         [0, 0, -1, 2, -1, 0, 0],
                         [0, 0, 0, -1, 2, -1, 0],
                         [0, 0, 0, 0, -1, 2, -1],
                         [0, 0, 0, 0, 0, -1, 2]])
    elif group_name == 'G2':
        # G2 Cartan matrix (2 x 2)
        return np.array([[2, -3],
                         [-1, 2]])
    else:
        raise ValueError(f"Unsupported group: {group_name}")


# --- Step 2: Generate Full Root System ---

def generate_full_root_system(cartan_matrix):
    rank = cartan_matrix.shape[0]
    simple_roots = []

    # Generate the simple root system from the Cartan matrix
    for i in range(rank):
        root = np.zeros(rank)
        root[i] = 1
        for j in range(i):
            root -= cartan_matrix[i, j] * simple_roots[j]
        simple_roots.append(root)

    # Construct all possible roots using linear combinations of simple roots
    root_system = set()
    combinations = np.array(list(itertools.product(range(-1, 2), repeat=rank)))

    for comb in combinations:
        root = np.dot(comb, simple_roots)
        if np.any(root):  # Ignore the zero vector
            root_system.add(tuple(root))

    root_system = np.array(list(root_system))

    return root_system


# --- Step 3: Construct Full Chevalley Basis for Lie Algebra ---

def construct_chevalley_basis(group_name):
    cartan_matrix = construct_cartan_matrix(group_name)
    rank = cartan_matrix.shape[0]

    # Cartan generators - diagonal matrices representing h_i
    cartan_generators = [np.diag([1 if i == j else 0 for j in range(rank)]) for i in range(rank)]

    # Constructing raising and lowering operators for each root
    positive_roots = generate_full_root_system(cartan_matrix)
    step_generators = []

    for root in positive_roots:
        if np.sum(root) > 0:  # Only positive roots
            e_alpha = construct_step_operator(root, rank)
            e_minus_alpha = construct_step_operator(-root, rank)
            step_generators.extend([e_alpha, e_minus_alpha])

    # Full Chevalley basis
    return cartan_generators + step_generators

def construct_step_operator(root, rank):
    """
    Construct a raising or lowering operator from a root vector.
    Uses an indexed approach to properly handle matrix construction.
    """
    step_operator = np.zeros((rank, rank))
    # Constructing non-trivial matrix corresponding to acting along a root direction
    for i in range(rank):
        if root[i] != 0:
            step_operator[i, (i + 1) % rank] = root[i]
    return step_operator


# --- Step 4: Generate Group Elements by Exponentiation ---

def generate_group_elements(basis_generators, num_samples):
    """
    Uses exponentiation of linear combinations of Lie algebra elements to generate group elements.
    """
    samples = []
    for _ in range(num_samples):
        coeffs = np.random.randn(len(basis_generators)) * 0.1  # Random small coefficients
        linear_combo = sum(coeffs[i] * basis_generators[i] for i in range(len(basis_generators)))
        group_element = expm(linear_combo)
        samples.append(group_element)
    return samples


# --- Step 5: Generate Conjugacy Classes ---

def generate_conjugacy_classes(samples):
    """
    Generates conjugacy classes by random conjugations of each sample.
    """
    conjugacy_classes = []
    for sample in samples:
        class_elements = [sample]
        for _ in range(50):  # More samples to properly generate the conjugacy class
            random_transform = np.random.randn(*sample.shape) * 0.01  # Small perturbations to stabilize
            random_transform_inv = np.linalg.pinv(random_transform)  # Use pseudo-inverse to avoid numerical issues
            conj = random_transform @ sample @ random_transform_inv
            class_elements.append(conj)
        conjugacy_classes.extend(class_elements)
    return conjugacy_classes


# --- Step 6: Calculate Symmetry Orbit Entropy (SOE) ---

def calculate_entropy(conjugacy_classes):
    """
    Calculate Shannon entropy of eigenvalue phases of elements in conjugacy classes.
    """
    eigenvalue_phases = []
    for matrix in conjugacy_classes:
        eigenvalues = np.linalg.eigvals(matrix)
        phases = np.angle(eigenvalues)
        eigenvalue_phases.extend(phases)

    eigenvalue_phases = np.array(eigenvalue_phases).reshape(-1, 1)

    # Use a more granular histogram to improve accuracy
    hist, bin_edges = np.histogram(eigenvalue_phases, bins=500, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist) * (bin_edges[1] - bin_edges[0]))

    return entropy


# --- Step 7: Main Function to Execute Full Workflow for Exceptional Groups ---

def main():
    num_samples = 10000  # Higher sample size for more statistical robustness

    groups = ['E8', 'F4', 'E6', 'E7', 'G2']  # Focus on various exceptional groups

    for group_name in groups:
        print(f"Calculating SOE for {group_name}...")

        # Step 1: Construct Chevalley Basis Generators
        basis_generators = construct_chevalley_basis(group_name)

        # Step 2: Generate Group Elements
        samples = generate_group_elements(basis_generators, num_samples)

        # Step 3: Generate Conjugacy Classes
        conjugacy_classes = generate_conjugacy_classes(samples)

        # Step 4: Calculate Entropy
        entropy = calculate_entropy(conjugacy_classes)

        print(f"Symmetry Orbit Entropy for {group_name}: {entropy}")


if __name__ == "__main__":
    main()
