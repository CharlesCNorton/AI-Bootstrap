import numpy as np
from scipy.integrate import nquad
from scipy.stats import sem, ttest_ind
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LieGroupType(Enum):
    CLASSICAL = "classical"
    EXCEPTIONAL = "exceptional"

@dataclass
class LieGroup:
    name: str
    rank: int
    dimension: int
    type: LieGroupType
    root_system: str

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate group parameters match known values"""
        if self.type == LieGroupType.EXCEPTIONAL:
            expected_dims = {
                "G2": 14,
                "F4": 52,
                "E6": 78,
                "E7": 133,
                "E8": 248
            }
            if self.name in expected_dims:
                assert self.dimension == expected_dims[self.name], f"Dimension mismatch for {self.name}"

def create_lie_group_catalog() -> Dict[str, LieGroup]:
    """Create catalog of all relevant Lie groups with exact parameters"""
    catalog = {}

    # Classical Series (up to rank 8)
    for n in range(1, 9):
        # A_n series (SU(n+1))
        catalog[f"A{n}"] = LieGroup(
            name=f"A{n}",
            rank=n,
            dimension=n*(n+2),
            type=LieGroupType.CLASSICAL,
            root_system="A"
        )

        # B_n series (SO(2n+1))
        if n >= 2:
            catalog[f"B{n}"] = LieGroup(
                name=f"B{n}",
                rank=n,
                dimension=n*(2*n+1),
                type=LieGroupType.CLASSICAL,
                root_system="B"
            )

        # C_n series (Sp(n))
        if n >= 3:
            catalog[f"C{n}"] = LieGroup(
                name=f"C{n}",
                rank=n,
                dimension=n*(2*n+1),
                type=LieGroupType.CLASSICAL,
                root_system="C"
            )

        # D_n series (SO(2n))
        if n >= 4:
            catalog[f"D{n}"] = LieGroup(
                name=f"D{n}",
                rank=n,
                dimension=n*(2*n-1),
                type=LieGroupType.CLASSICAL,
                root_system="D"
            )

    # Exceptional Groups
    exceptional_groups = [
        ("G2", 2, 14),
        ("F4", 4, 52),
        ("E6", 6, 78),
        ("E7", 7, 133),
        ("E8", 8, 248)
    ]

    for name, rank, dim in exceptional_groups:
        catalog[name] = LieGroup(
            name=name,
            rank=rank,
            dimension=dim,
            type=LieGroupType.EXCEPTIONAL,
            root_system=name[0]
        )

    return catalog

def compute_root_system_density(group: LieGroup, x: np.ndarray) -> np.ndarray:
    """
    Compute density based on actual root system structure
    """
    norm = np.sqrt(np.sum(x**2, axis=-1))

    if group.type == LieGroupType.EXCEPTIONAL:
        if group.name == "E8":
            # E8 has 240 roots and highly symmetric structure
            return np.exp(-norm/8) * (1 + 0.5*np.cos(norm*np.pi/2))
        elif group.name == "E7":
            # E7 has 126 roots
            return np.exp(-norm/7) * (1 + 0.4*np.cos(norm*np.pi/2))
        elif group.name == "E6":
            # E6 has 72 roots
            return np.exp(-norm/6) * (1 + 0.3*np.cos(norm*np.pi/2))
        elif group.name == "F4":
            # F4 has 48 roots of two different lengths
            return np.exp(-norm/4) * (1 + 0.25*np.cos(norm*np.pi/2))
        elif group.name == "G2":
            # G2 has 12 roots of two different lengths
            return np.exp(-norm/2) * (1 + 0.2*np.cos(norm*np.pi/2))
    else:
        # Classical groups have simpler root systems
        if group.root_system == "A":
            return np.exp(-norm/group.rank)
        elif group.root_system == "B":
            return np.exp(-norm/group.rank) * (1 + 0.1*np.cos(norm*np.pi/2))
        elif group.root_system == "C":
            return np.exp(-norm/group.rank) * (1 + 0.15*np.cos(norm*np.pi/2))
        elif group.root_system == "D":
            return np.exp(-norm/group.rank) * (1 + 0.2*np.cos(norm*np.pi/2))

    raise ValueError(f"Unknown root system for group {group.name}")

def compute_entropy(group: LieGroup, num_samples: int = 10000, num_bootstrap: int = 100) -> Tuple[float, float]:
    """
    Compute entropy with error estimation using bootstrap resampling
    """
    try:
        # Generate points in root space
        rng = np.random.default_rng()
        points = rng.normal(0, 1, (num_samples, group.rank))

        # Compute densities
        densities = compute_root_system_density(group, points)

        # Normalize densities
        densities = densities / np.sum(densities)

        # Bootstrap for error estimation
        entropies = []
        for _ in range(num_bootstrap):
            idx = rng.integers(0, num_samples, num_samples)
            boot_densities = densities[idx]
            boot_densities = boot_densities / np.sum(boot_densities)
            entropy = -np.sum(boot_densities * np.log(boot_densities + 1e-10))
            entropies.append(entropy)

        return np.mean(entropies), np.std(entropies)

    except Exception as e:
        logger.error(f"Error computing entropy for {group.name}: {str(e)}")
        raise

def run_comprehensive_test():
    """
    Run comprehensive test across all Lie groups
    """
    try:
        catalog = create_lie_group_catalog()
        results = {}
        errors = {}

        # Compute entropies with error bars
        for name, group in catalog.items():
            logger.info(f"Computing entropy for {name}")
            entropy, error = compute_entropy(group)
            results[name] = entropy
            errors[name] = error

            logger.info(f"{name}: S = {entropy:.4f} ± {error:.4f}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

        # Plot by rank
        ranks = range(1, 9)
        classical_by_rank = {r: [] for r in ranks}
        exceptional_by_rank = {r: [] for r in ranks}

        for name, entropy in results.items():
            group = catalog[name]
            if group.type == LieGroupType.CLASSICAL:
                classical_by_rank[group.rank].append(entropy)
            else:
                exceptional_by_rank[group.rank].append(entropy)

        # Compute means and errors for classical groups by rank
        classical_means = [np.mean(classical_by_rank[r]) if classical_by_rank[r] else np.nan for r in ranks]
        classical_errors = [np.std(classical_by_rank[r]) if len(classical_by_rank[r]) > 1 else 0 for r in ranks]

        # Plot with error bars
        ax1.errorbar(ranks, classical_means, yerr=classical_errors, label='Classical Groups', fmt='o-')

        # Plot exceptional groups individually
        exceptional_groups = [g for g in catalog.values() if g.type == LieGroupType.EXCEPTIONAL]
        ex_ranks = [g.rank for g in exceptional_groups]
        ex_entropies = [results[g.name] for g in exceptional_groups]
        ex_errors = [errors[g.name] for g in exceptional_groups]

        ax1.errorbar(ex_ranks, ex_entropies, yerr=ex_errors, label='Exceptional Groups', fmt='s-')

        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy vs Rank for Classical and Exceptional Lie Groups')
        ax1.legend()
        ax1.grid(True)

        # Plot by dimension
        dims_classical = [g.dimension for g in catalog.values() if g.type == LieGroupType.CLASSICAL]
        dims_exceptional = [g.dimension for g in catalog.values() if g.type == LieGroupType.EXCEPTIONAL]

        entropies_classical = [results[g.name] for g in catalog.values() if g.type == LieGroupType.CLASSICAL]
        entropies_exceptional = [results[g.name] for g in catalog.values() if g.type == LieGroupType.EXCEPTIONAL]

        errors_classical = [errors[g.name] for g in catalog.values() if g.type == LieGroupType.CLASSICAL]
        errors_exceptional = [errors[g.name] for g in catalog.values() if g.type == LieGroupType.EXCEPTIONAL]

        ax2.errorbar(dims_classical, entropies_classical, yerr=errors_classical,
                    fmt='o', label='Classical Groups')
        ax2.errorbar(dims_exceptional, entropies_exceptional, yerr=errors_exceptional,
                    fmt='s', label='Exceptional Groups')

        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy vs Dimension for Lie Groups')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Statistical analysis
        classical_values = np.array([v for k, v in results.items()
                                   if catalog[k].type == LieGroupType.CLASSICAL])
        exceptional_values = np.array([v for k, v in results.items()
                                     if catalog[k].type == LieGroupType.EXCEPTIONAL])

        # Perform t-test
        t_stat, p_value = ttest_ind(classical_values, exceptional_values)

        return {
            'results': results,
            'errors': errors,
            't_statistic': t_stat,
            'p_value': p_value
        }

    except Exception as e:
        logger.error(f"Error in comprehensive test: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    try:
        results = run_comprehensive_test()
        end_time = time.time()

        logger.info(f"\nTest completed in {end_time - start_time:.2f} seconds")
        logger.info(f"T-statistic: {results['t_statistic']:.4f}")
        logger.info(f"P-value: {results['p_value']:.4f}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
